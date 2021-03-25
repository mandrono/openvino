// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_graph_optimizer.h"

#include "mkldnn_extension_utils.h"
#include "nodes/mkldnn_reshape_node.h"
#include "nodes/mkldnn_pooling_node.h"
#include "nodes/mkldnn_eltwise_node.h"
#include "nodes/mkldnn_concat_node.h"
#include "nodes/mkldnn_reorder_node.h"
#include "nodes/mkldnn_conv_node.h"
#include "nodes/mkldnn_bin_conv_node.h"
#include "nodes/mkldnn_fake_quantize_node.h"
#include "nodes/mkldnn_mvn_node.h"
#include <nodes/mkldnn_transpose_node.h>
#include "nodes/mkldnn_interpolate_node.h"
#include "nodes/mkldnn_input_node.h"
#include "nodes/common/cpu_convert.h"

#include "mkldnn/ie_mkldnn.h"

#include <blob_factory.hpp>
#include "utils/general_utils.h"

// WA for xbyak.h
#ifdef _WIN32
# ifndef _WINSOCKAPI_
#  define _WINSOCKAPI_
# endif
# ifndef _WINSOCK2API_
#  define _WINSOCK2API_
#endif
#endif
#include <cpu/x64/cpu_isa_traits.hpp>

#include <string>
#include <list>
#include <memory>
#include <set>
#include <algorithm>

#include "mkldnn_itt.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGraphOptimizer::MKLDNNGraphOptimizer() {}

void MKLDNNGraphOptimizer::ApplyCommonGraphOptimizations(MKLDNNGraph &graph) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraphOptimizer::ApplyCommonGraphOptimizations");
// TODO [NM]: transformation should be implemented w/o using of CNNLayer
//    MergeTwoEqualScaleShifts(graph);
//    graph.RemoveDroppedNodes();

    FuseConvolutionAndBias(graph);
    graph.RemoveDroppedNodes();

    FuseMultiplyAndAdd(graph);
    graph.RemoveDroppedNodes();

    FuseDeconvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseBroadcastAndEltwise(graph);
    graph.RemoveDroppedNodes();

    FuseClampAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

    FuseMulAddAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

// TODO [NM]: transformation should be implemented w/o using of CNNLayer
//    FuseConvolutionAndZeroPoints(graph);
//    graph.RemoveDroppedNodes();

// TODO [NM]: While fusing simple operation into any node (except Eltwise) we need to check that other inputs are Constant nodes.
    FuseConvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    graph.SortTopologically();
    graph.RemoveDroppedEdges();

    FusePoolingAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

    graph.SortTopologically();
    graph.RemoveDroppedEdges();

// TODO [NM]: transformation should be implemented w/o using of CNNLayer
//    FuseConvolutionAndDWConvolution(graph);
//    graph.RemoveDroppedNodes();

    FuseBinaryConvolutionAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionSumAndConvolutionSumActivation(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseFullyConnectedAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseMVNAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseInterpolateAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseNormalizeL2AndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseEltwiseAndSimple(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations(MKLDNNGraph &graph) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations");

    DropDoubleReorders(graph);
    graph.RemoveDroppedNodes();

#if 0
    /* disable, since there is no use case for it at the moment
     * should be enabled after ngraph migration */
    DropConvertReorder(graph);
    graph.RemoveDroppedNodes();
#endif

    MergeTransposeAndReorder(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::FuseConvolutionAndBias(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Convolution &&
               node->getChildEdges().size() == 1 &&
               node->getFusedWith().empty();
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if ((parentNode->isConstant() && !childNode->isConstant()) || childNode->getAlgorithm() != EltwiseAdd || !childNode->getFusedWith().empty() ||
            childNode->getParentEdges().size() != 2)
            return false;

        auto biasNode = childNode->getParentEdgesAtPort(1)[0]->getParent();
        if (biasNode->getChildEdges().size() != 1)
            return false;

        auto convOutDims = parentNode->getChildEdgesAtPort(0)[0]->getDims();
        auto biasDims = biasNode->getChildEdgesAtPort(0)[0]->getDims();
        // TODO [NM]: Legacy ConvBias fusion transformation supports both per-tensor (via explicit broadcasing) and per-channel cases.
        // Most of the real models contain per-channel bias, so we need to reavaluate the need to support per-tensor variant.
        if (convOutDims.ndims() != biasDims.ndims() || biasDims.ndims() < 2)
            return false;

        if (biasDims[0] != 1 || biasDims[1] != convOutDims[1])
            return false;

        for (int i = 2; i < biasDims.ndims(); i++) {
            if (biasDims[i] != 1)
                return false;
        }

        return true;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        auto childs = childNode->childEdges;
        auto parents = childNode->parentEdges;

        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (!p_edge) continue;
            auto parent = p_edge->getParent();
            if (!parent) continue;

            if (parent == parentNode) {
                for (size_t j = 0; j < childs.size(); j++) {
                    if (!childs[j].lock())
                        continue;
                    auto child = childs[j].lock()->getChild();
                    if (!child)
                        continue;

                    MKLDNNEdgePtr &remEdge = p_edge;
                    int inNum = 0;
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        remEdge->drop();
                        removeEdge(graph, remEdge);
                    }
                    remEdge = childs[j].lock();
                    int outNum = 0;
                    if (remEdge) {
                        outNum = remEdge->getOutputNum();
                        remEdge->drop();
                        removeEdge(graph, remEdge);
                    }
                    MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);
                }
            } else {
                MKLDNNEdgePtr &remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    remEdge->drop();
                    removeEdge(graph, remEdge);
                }

                auto parentEltwise = parentNode;
                MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
                auto &graphEdges = graph.GetEdges();
                graphEdges.push_back(newEdge);
                parent->addEdge(newEdge);

                auto newBiasDim = parent->outDims[inNum][1];
                parent->outDims[inNum] = MKLDNNDims({newBiasDim});
                parentEltwise->inDims.push_back(parent->outDims[0]);
            }
        }

        graph.DropNode(childNode);

        parentNode->addOriginalInputPrecision(childNode->getOriginalInputPrecisionAtPort(1));
    }
}

void MKLDNNGraphOptimizer::FuseDeconvolutionAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Deconvolution && node->getChildEdges().size() == 1 && node->getFusedWith().empty();
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        // at this moment deconvolution supports only depthwise as post op
        if (!childNode->canBePerformedAsScaleShift(parentNode.get())) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        auto parentEdges = childNode->parentEdges;
        for (auto &parentEdge : parentEdges) {
            auto p_edge = parentEdge.lock();
            if (p_edge->getParent()->getType() == Deconvolution)
                continue;

            removeEdge(graph, p_edge);
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseMultiplyAndAdd(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableSecondInput = [](MKLDNNNodePtr node, MKLDNNDims dataDims) {
        auto secondInputDims = node->outDims[0];
        if (secondInputDims.ndims() != dataDims.ndims() || secondInputDims.ndims() < 2)
            return false;

        if (secondInputDims[0] != 1 || secondInputDims[1] != dataDims[1])
            return false;

        for (size_t i = 2; i < secondInputDims.ndims(); i++) {
            if (secondInputDims[i] != 1)
                return false;
        }

        return true;
    };

    auto isSutableParentNode = [&](MKLDNNNodePtr node) {
        if (node->getAlgorithm() != EltwiseMultiply || !node->getFusedWith().empty() ||
            node->getParentEdges().size() != 2 || node->getChildEdges().size() != 1)
            return false;

        return isSutableSecondInput(node->getParentEdgesAtPort(1)[0]->getParent(), node->getParentEdgesAtPort(0)[0]->getDims());
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if ((parentNode->isConstant() && !childNode->isConstant()) || childNode->getAlgorithm() != EltwiseAdd || !childNode->getFusedWith().empty() ||
            childNode->getParentEdges().size() != 2)
            return false;

        return isSutableSecondInput(childNode->getParentEdgesAtPort(1)[0]->getParent(), childNode->getParentEdgesAtPort(0)[0]->getDims());
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        auto childs = childNode->childEdges;
        auto parents = childNode->parentEdges;

        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (!p_edge) continue;
            auto parent = p_edge->getParent();
            if (!parent) continue;

            if (parent == parentNode) {
                for (size_t j = 0; j < childs.size(); j++) {
                    if (!childs[j].lock())
                        continue;
                    auto child = childs[j].lock()->getChild();
                    if (!child)
                        continue;

                    MKLDNNEdgePtr &remEdge = p_edge;
                    int inNum = 0;
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        remEdge->drop();
                        removeEdge(graph, remEdge);
                    }
                    remEdge = childs[j].lock();
                    int outNum = 0;
                    if (remEdge) {
                        outNum = remEdge->getOutputNum();
                        remEdge->drop();
                        removeEdge(graph, remEdge);
                    }
                    MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);
                }
            } else {
                MKLDNNEdgePtr &remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    remEdge->drop();
                    removeEdge(graph, remEdge);
                }

                auto parentEltwise = parentNode;
                MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
                auto &graphEdges = graph.GetEdges();
                graphEdges.push_back(newEdge);
                parent->addEdge(newEdge);

                parentEltwise->inDims.push_back(parent->outDims[0]);
            }
        }

        parentNode->addOriginalInputPrecision(childNode->getOriginalInputPrecisionAtPort(1));
        parentNode->setAlgorithm(EltwiseMulAdd);
        parentNode->addOriginalLayer(childNode->getOriginalLayers());

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndZeroPoints(MKLDNNGraph &graph) {
//    auto& graphNodes = graph.GetNodes();
//
//    auto isSutableConvNode = [](MKLDNNNodePtr node) {
//        if (node->getType() != Convolution)
//            return false;
//
//        if (node->getParentEdges().size() < 2)
//            return false;
//
//        auto* convLayer = dynamic_cast<ConvolutionLayer*>(node->getCnnLayer().get());
//        if (convLayer == nullptr)
//            IE_THROW() << "Cannot get convolution layer " << node->getName();
//
//        return true;
//    };
//
//    auto initializeInputZeroPoints = [](MKLDNNNodePtr node, MKLDNNNodePtr parent0, MKLDNNNodePtr parent1) {
//        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
//        if (convNode == nullptr)
//            IE_THROW() << "Cannot get convolution node " << node->getName();
//
//        int IC = node->getParentEdgesAtPort(0)[0]->getDims()[1];
//        int OC = node->getChildEdgesAtPort(0)[0]->getDims()[1];
//
//        if (parent0->getType() == Eltwise) {
//            // The plug-in doesn't support FP32 convolution with input/weights zero points.
//            // In case weights are in FP32 (or we have zero points on weights which are not supported by INT8 convolution) we cannot use
//            // INT8 implementation so we have to disable input zero points fusing as well.
//            auto weightsLayer = parent1->getCnnLayer();
//            if (!weightsLayer || weightsLayer->type != "Const" || weightsLayer->outData[0]->getPrecision() != Precision::I8) {
//                return false;
//            }
//
//            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent0.get());
//            if (eltwiseNode->getOpType() != Subtract)
//                return false;
//
//            if (parent0->getParentEdges().size() != 2)
//                return false;
//
//            if (parent0->getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->type == "Const") {
//                auto arg0 = parent0->getParentEdgesAtPort(1)[0]->getParent();
//                if (arg0->getCnnLayer()->outData[0]->getPrecision() != Precision::U8)
//                    return false;
//
//                if (parent0->getParentEdgesAtPort(1)[0]->getDims().size() < 2) {
//                    return false;
//                }
//
//                if (parent0->getParentEdgesAtPort(1)[0]->getDims()[1] != 1 &&
//                    parent0->getParentEdgesAtPort(1)[0]->getDims()[1] != IC)
//                    return false;
//
//                auto arg1 = parent0->getParentEdgesAtPort(0)[0]->getParent();
//                if (arg1->getCnnLayer()->outData[0]->getPrecision() != Precision::U8)
//                    return false;
//
//                auto zeroPointsBlob = dynamic_cast<TBlob<uint8_t>*>(arg0->getCnnLayer()->blobs["custom"].get());
//                if (zeroPointsBlob == nullptr)
//                    IE_THROW() << "Cannot cast to TBlob internal zero points blob";
//
//                auto zeroPointsData = zeroPointsBlob->buffer().as<uint8_t*>();
//                if (zeroPointsData == nullptr)
//                    IE_THROW() << "zeroPointsBlob has not allocated buffer";
//
//                for (int j = 0; j < parent0->getParentEdgesAtPort(1)[0]->getDims()[1]; j++) {
//                    convNode->inputZeroPoints.push_back(zeroPointsData[j]);
//                }
//            } else {
//                return false;
//            }
//        } else {
//            return false;
//        }
//
//        if (convNode->outputCompensation.empty()) {
//            convNode->outputCompensation.resize(OC);
//        }
//
//        return true;
//    };
//
////    auto initializeWeightsZeroPoints = [](MKLDNNNodePtr node, MKLDNNNodePtr parent0) {
////        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
////        if (convNode == nullptr)
////            IE_THROW() << "Cannot get convolution node " << node->getName();
////
////        int OC = node->getChildEdgesAtPort(0)[0]->getDims()[1];
////
////        if (parent0->getType() == Eltwise) {
////            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent0.get());
////            if (eltwiseNode->getOpType() != Subtract)
////                return false;
////
////            if (parent0->getParentEdges().size() != 2)
////                return false;
////
////            if (parent0->getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->type == "Const") {
////                auto arg0 = parent0->getParentEdgesAtPort(1)[0]->getParent();
////                if (arg0->getCnnLayer()->outData[0]->getPrecision() != Precision::I8)
////                    return false;
////
////                if (parent0->getParentEdgesAtPort(1)[0]->getDims()[0] != 1 &&
////                    parent0->getParentEdgesAtPort(1)[0]->getDims()[0] != OC)
////                    return false;
////
////                auto arg1 = parent0->getParentEdgesAtPort(0)[0]->getParent();
////                if (arg1->getCnnLayer()->outData[0]->getPrecision() != Precision::I8)
////                    return false;
////
////                auto zeroPointsBlob = dynamic_cast<TBlob<int8_t>*>(arg0->getCnnLayer()->blobs["custom"].get());
////                if (zeroPointsBlob == nullptr)
////                    IE_THROW() << "Cannot cast to TBlob internal zero points blob";
////
////                auto zeroPointsData = zeroPointsBlob->buffer().as<int8_t*>();
////                if (zeroPointsData == nullptr)
////                    IE_THROW() << "zeroPointsBlob has not allocated buffer";
////
////                for (int j = 0; j < parent0->getParentEdgesAtPort(1)[0]->getDims()[0]; j++) {
////                    convNode->weightsZeroPoints.push_back(static_cast<float>(zeroPointsData[j]));
////                }
////            } else {
////                return false;
////            }
////        } else {
////            return false;
////        }
////
////        return true;
////    };
//
//    auto initializeOutputCompensation = [](MKLDNNNodePtr node) {
//        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
//        if (convNode == nullptr)
//            IE_THROW() << "Cannot get convolution node " << node->getName();
//
//        auto * convLayer = dynamic_cast<ConvolutionLayer*>(convNode->getCnnLayer().get());
//        if (convLayer == nullptr)
//            IE_THROW() << "Cannot get eltwise layer " << node->getName();
//
//        for (int i = 0; i < convLayer->insData.size(); i++)
//            if (convLayer->insData[i].lock() == nullptr)
//                IE_THROW() << "Node '"<< node->getName() << "' has invalid input data with index " << i;
//
//        if (convNode->inputZeroPoints.empty())
//            return;
//
//        auto weightsLayer = getCreatorLayer(convLayer->insData[1].lock()).lock();
//        if (weightsLayer->type != "Const") {
//            weightsLayer = getCreatorLayer(weightsLayer->insData[0].lock()).lock();
//        }
//
//
//        auto weightsBlob = dynamic_cast<TBlob<int8_t>*>(weightsLayer->blobs["custom"].get());
//        if (weightsBlob == nullptr)
//            IE_THROW() << "Cannot cast to TBlob internal weights blob";
//
//        auto weightsPtr = weightsBlob->buffer().as<int8_t*>();
//        if (weightsPtr == nullptr)
//            IE_THROW() << "weightsBlob has not allocated buffer";
//
//        ptrdiff_t G = convLayer->_group;
//        ptrdiff_t OC = weightsLayer->outData[0]->getDims()[0] / G;
//        ptrdiff_t IC = weightsLayer->outData[0]->getDims()[1];
//        ptrdiff_t KD = weightsLayer->outData[0]->getDims().size() == 5 ? weightsLayer->outData[0]->getDims()[2] : 1;
//        ptrdiff_t KH = weightsLayer->outData[0]->getDims()[weightsLayer->outData[0]->getDims().size() - 2];
//        ptrdiff_t KW = weightsLayer->outData[0]->getDims()[weightsLayer->outData[0]->getDims().size() - 1];
//
//        for (size_t g = 0; g < G; g++) {
//            for (size_t oc = 0; oc < OC; oc++) {
//                int32_t a = 0;
//                for (size_t ic = 0; ic < IC; ic++) {
//                    for (size_t kd = 0; kd < KD; kd++) {
//                        for (size_t kh = 0; kh < KH; kh++) {
//                            for (size_t kw = 0; kw < KW; kw++) {
//                                size_t widx = g * OC * IC * KD * KH * KW +
//                                              oc * IC * KD * KH * KW +
//                                              ic * KD * KH * KW +
//                                              kd * KH * KW +
//                                              kh * KW +
//                                              kw;
//
//                                auto w = static_cast<int32_t>(weightsPtr[widx]);
//
//                                auto izp = !convNode->inputZeroPoints.empty() ? static_cast<int32_t>(convNode->inputZeroPoints[g * IC + ic]) : 0;
//                                a += w * izp;
//
//                                auto wzp = !convNode->weightsZeroPoints.empty() ? static_cast<int32_t>(convNode->weightsZeroPoints[g * OC + oc]) : 0;
//                                a -= wzp * izp;
//                            }
//                        }
//                    }
//                }
//                convNode->outputCompensation[g * OC + oc] = -a;
//            }
//        }
//    };
//
//    for (int i = 0; i < graphNodes.size(); i++) {
//        auto conv = graphNodes[i];
//        if (!isSutableConvNode(conv)) continue;
//
//        auto dataEltwise = conv->getParentEdgesAtPort(0)[0]->getParent();
//        auto weightsEltwise = conv->getParentEdgesAtPort(1)[0]->getParent();
//        if (initializeInputZeroPoints(conv, dataEltwise, weightsEltwise)) {
//            auto p_edge = dataEltwise->getParentEdgesAtPort(1)[0];
//            removeEdge(graph, p_edge);
//
//            graph.DropNode(dataEltwise);
//        }
//
//// [TODO] Weights zero point is not supported on oneDNN side for the moment
////        auto weightsEltwise = conv->getParentEdgesAtPort(1)[0]->getParent();
////        if (initializeWeightsZeroPoints(conv, weightsEltwise)) {
////            auto p_edge = weightsEltwise->getParentEdgesAtPort(1)[0];
////            removeEdge(graph, p_edge);
////
////            graph.DropNode(weightsEltwise);
////        }
//
//        initializeOutputCompensation(conv);
//    }
}

//  WA: We need it until LP transformations will not optimize this pattern inside
void MKLDNNGraphOptimizer::MergeTwoEqualScaleShifts(MKLDNNGraph& graph) {
//    auto& graphNodes = graph.GetNodes();
//
//    auto isSutableScaleShiftNode = [](MKLDNNNodePtr node) {
//        if (node->getType() != Eltwise)
//            return false;
//
//        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
//        if (eltwiseNode == nullptr)
//            IE_THROW() << "Cannot cast " << node->getName() << " to Eltwise node";
//
//        if (eltwiseNode->getChildEdges().size() != 1)
//            return false;
//
//        if (eltwiseNode->getOpType() != MulAdd)
//            return false;
//
//        return true;
//    };
//
//    auto isEqualScaleShiftNodes = [](MKLDNNNodePtr node1, MKLDNNNodePtr node2) {
//        if (node1->getParentEdgeAt(0) != node2->getParentEdgeAt(0))
//            return false;
//
//        auto *eltwiseNode1 = dynamic_cast<MKLDNNEltwiseNode *>(node1.get());
//        auto *eltwiseNode2 = dynamic_cast<MKLDNNEltwiseNode *>(node2.get());
//
//        auto eltwiseLayer1 = eltwiseNode1->getCnnLayer();
//        auto eltwiseLayer2 = eltwiseNode2->getCnnLayer();
//
//        Blob::Ptr scalesBlob1 = eltwiseLayer1->blobs["weights"];
//        Blob::Ptr shiftsBlob1 = eltwiseLayer1->blobs["biases"];
//        Blob::Ptr scalesBlob2 = eltwiseLayer2->blobs["weights"];
//        Blob::Ptr shiftsBlob2 = eltwiseLayer2->blobs["biases"];
//        if (scalesBlob1 == nullptr || shiftsBlob1 == nullptr || scalesBlob2 == nullptr || shiftsBlob2 == nullptr)
//            return false;
//
//        if (scalesBlob1->size() != shiftsBlob1->size() || scalesBlob2->size() != shiftsBlob2->size()
//            || scalesBlob1->size() != scalesBlob2->size()) return false;
//
//        const float *scalesBufferPtr1 = scalesBlob1->buffer().as<float *>();
//        const float *shiftsBufferPtr1 = shiftsBlob1->buffer().as<float *>();
//        const float *scalesBufferPtr2 = scalesBlob2->buffer().as<float *>();
//        const float *shiftsBufferPtr2 = shiftsBlob2->buffer().as<float *>();
//
//        for (int i = 0; i < scalesBlob1->size(); i++)
//            if (scalesBufferPtr1[i] != scalesBufferPtr2[i] || shiftsBufferPtr1[i] != shiftsBufferPtr2[i])
//                return false;
//
//        return true;
//    };
//
//    auto MergeScaleShiftNodes = [&](MKLDNNNodePtr childNode1, MKLDNNNodePtr childNode2) {
//        auto parentNode = childNode2->getParentEdgeAt(0)->getParent();
//        auto ccNode2 = childNode2->getChildEdgeAt(0)->getChild();
//
//        auto parentEdges = childNode2->parentEdges;
//        for (auto &parentEdge : parentEdges) {
//            auto p_edge = parentEdge.lock();
//            if (p_edge->getParent() == parentNode)
//                continue;
//
//            removeEdge(graph, p_edge);
//        }
//
//        graph.DropNode(childNode2);
//
//        MKLDNNEdgePtr remEdge;
//        for (auto edge : parentNode->getChildEdges()) {
//            if (edge.lock()->getChild() == ccNode2) {
//                remEdge = edge.lock();
//                break;
//            }
//        }
//        if (remEdge == nullptr)
//            IE_THROW() << "Edge was not found";
//        remEdge->drop();
//        graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), remEdge), graph.GetEdges().end());
//
//        if (childNode1->getChildEdgeAt(0)->getChild() != ccNode2) {
//            auto iIndex = childNode1->getChildEdgeAt(0)->getInputNum();
//            auto oIndex = remEdge->getOutputNum();
//            MKLDNNEdgePtr newEdge(new MKLDNNEdge(childNode1, ccNode2, iIndex, oIndex));
//            childNode1->addEdge(newEdge);
//            graph.GetEdges().push_back(newEdge);
//        }
//    };
//
//    for (int i = 0; i < graphNodes.size(); i++) {
//        auto parentNode = graphNodes[i];
//        if (parentNode->getChildEdges().size() != 2) continue;
//
//        auto childNode1 = parentNode->getChildEdgeAt(0)->getChild();
//        if (!isSutableScaleShiftNode(childNode1)) continue;
//
//        auto childNode2 = parentNode->getChildEdgeAt(1)->getChild();
//        if (!isSutableScaleShiftNode(childNode2)) continue;
//
//        if (!isEqualScaleShiftNodes(childNode1, childNode2)) continue;
//
//        MergeScaleShiftNodes(childNode1, childNode2);
//    }
}

void MKLDNNGraphOptimizer::FuseFullyConnectedAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == FullyConnected && node->getChildEdges().size() == 1 && node->getParentEdgeAt(0)->getDims().ndims() != 3;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuseSimpleOperation(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == FullyConnected)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndDWConvolution(MKLDNNGraph &graph) {
//    auto& graphNodes = graph.GetNodes();
//
//    auto isConvolutionNode = [](MKLDNNNodePtr node) {
//        return node->getType() == Convolution;
//    };
//
//    auto is1x1Convolution = [](ConvolutionLayer* layer) {
//        return layer->_kernel[X_AXIS] == 1 && layer->_kernel[Y_AXIS] == 1;
//    };
//
//    auto isSutableParentConvolution = [&](MKLDNNNodePtr node) {
//        auto *layer = dynamic_cast<ConvolutionLayer *>(node->getCnnLayer().get());
//        if (layer == nullptr)
//            IE_THROW() << "Cannot get convolution layer " << node->getName();
//
//        auto* parentConvolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
//        if (parentConvolutionNode == nullptr)
//            IE_THROW() << "Cannot get convolution node " << node->getName();
//
//        if (!parentConvolutionNode->weightsZeroPoints.empty())
//            return false;
//
//        // TODO [oneDNN]: is it still valide constrain on conv to fuse in?
//        bool isSupportedParams = layer->_group == 1 &&
//                is1x1Convolution(layer) &&  // TODO [oneDNN] : fusing is permitted only with 1x1 convolutions
//                everyone_is(1, layer->_stride[X_AXIS], layer->_stride[Y_AXIS]) &&
//                one_of(layer->outData[0].get()->getPrecision(), Precision::FP32, Precision::U8) &&
//                node->getChildEdgeAt(0)->getDims().ndims() == 4;
//        if (!isSupportedParams) return false;
//
//        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
//    };
//
//    auto isSutableChildConvolution = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
//        auto* childLayer = dynamic_cast<ConvolutionLayer*>(childNode->getCnnLayer().get());
//        if (childLayer == nullptr)
//            IE_THROW() << "Cannot get convolution layer " << childNode->getName();
//
//        auto* parentLayer = dynamic_cast<ConvolutionLayer*>(parentNode->getCnnLayer().get());
//        if (parentLayer == nullptr)
//            IE_THROW() << "Cannot get convolution layer " << parentNode->getName();
//
//        if (parentLayer->outData[0].get()->getPrecision() != childLayer->outData[0].get()->getPrecision())
//            return false;
//
//        if (parentLayer->precision != childLayer->precision)
//            return false;
//
//        auto parentOutputPrecision = !parentNode->fusedWith.empty()
//                ? parentNode->fusedWith[parentNode->fusedWith.size() - 1]->getCnnLayer()->outData[0].get()->getPrecision()
//                : parentNode->getCnnLayer()->outData[0].get()->getPrecision();
//
//        auto childOutputPrecision = !childNode->fusedWith.empty()
//                ? childNode->fusedWith[childNode->fusedWith.size() - 1]->getCnnLayer()->outData[0].get()->getPrecision()
//                : childNode->getCnnLayer()->outData[0].get()->getPrecision();
//
//        if (parentOutputPrecision != childOutputPrecision)
//            return false;
//
//        auto* childConvolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(childNode.get());
//        if (childConvolutionNode == nullptr)
//            IE_THROW() << "Cannot get convolution node " << childNode->getName();
//
//        if (!childConvolutionNode->inputZeroPoints.empty() || !childConvolutionNode->weightsZeroPoints.empty())
//            return false;
//
//        auto allPads = getPaddings(*childLayer);
//
//        bool isSupportedParams = childLayer->_out_depth == childLayer->_group &&
//                                 childLayer->_out_depth != 1 &&
//                                 everyone_is(3, childLayer->_kernel[X_AXIS], childLayer->_kernel[Y_AXIS]) &&
//                                 everyone_is(1, allPads.begin[X_AXIS], allPads.begin[Y_AXIS]) &&
//                                 everyone_is(1, allPads.end[X_AXIS], allPads.end[Y_AXIS]) &&
//                                 everyone_is(1, childLayer->_dilation[X_AXIS], childLayer->_dilation[Y_AXIS]) &&
//                                 childLayer->_stride[X_AXIS] == childLayer->_stride[Y_AXIS] &&
//                                 false &&  // TODO [oneDNN]: disabled while not ported
//                                 one_of(childLayer->_stride[X_AXIS], 1 /*, 2*/) &&  // TODO [oneDNN]: stride 2 should also be supported
//                                 childNode->getChildEdgeAt(0)->getDims().ndims() == 4;
//
//        return isSupportedParams;
//    };
//
//    for (int i = 0; i < graphNodes.size(); i++) {
//        if (!isConvolutionNode(graphNodes[i])) continue;
//
//        auto parentConvNode = graphNodes[i];
//        if (!isSutableParentConvolution(parentConvNode)) continue;
//
//        auto childConvNode = parentConvNode->getChildEdgeAt(0)->getChild();
//        if (!isSutableChildConvolution(parentConvNode, childConvNode)) continue;
//
//        parentConvNode->fuseWith(childConvNode);
//
//        for (auto node : childConvNode->getFusedWith())
//            parentConvNode->fuseWith(node);
//        childConvNode->clearFusedWith();
//
//        graph.DropDWConvNode(childConvNode);
//    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Convolution && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuseSimpleOperation(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Convolution)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseBinaryConvolutionAndFakeQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == BinaryConvolution && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if ((parentNode->isConstant() && !childNode->isConstant()) || childNode->getType() != FakeQuantize)
            return false;

        auto* binConv = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parentNode.get());
        if (!binConv) {
            return false;
        }

        return binConv->canFuse(childNode);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parent, child)) continue;

        child->fuseInto(parent);

        auto parents = child->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == BinaryConvolution)
                continue;

            removeEdge(graph, p_edge);
        }

        graph.DropNode(child);
    }
}

void MKLDNNGraphOptimizer::FusePoolingAndFakeQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Pooling && node->getChildEdges().size() == 1 && node->getAlgorithm() == Algorithm::PoolingAvg;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        return node->getType() == FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(child)) continue;

        child->fuseInto(parent);

        auto parents = child->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == Pooling)
                continue;

            removeEdge(graph, p_edge);
        }

        graph.DropNode(child);
    }
}

/**
 *  Check if there is a data dependency between parent and child
 *  BFS starting from parent and comparing with child
 *
 * @param parent head of BFS
 * @param child node we try to find
 * @return True if child is one of data supplier
 */
static bool is_data_dependency(const std::shared_ptr<MKLDNNNode> &parent,
                               const std::shared_ptr<MKLDNNNode> &child) {
    std::set<MKLDNNNode*> visited;
    std::list<MKLDNNNode*> nextLayers {parent.get()};

    for (; !nextLayers.empty();) {
        auto layer = *nextLayers.begin();
        if (layer == child.get()) return true;
        for (auto oe : layer->getChildEdges()) {
            auto nn = oe.lock()->getChild();
            if (visited.find(nn.get()) == visited.end()) {
                nextLayers.push_back(nn.get());
                visited.insert(nn.get());
            }
        }
        nextLayers.pop_front();
    }
    return false;
}

/*
 *  Before:
 *
 *        ***             ***                   ***             ***
 *         |               |                     |               |
 *    +========+       +========+           +========+       +========+
 *    |  any   |       | conv 2 |           |  any   |       | conv 2 |
 *    +========+       +========+           +========+       +========+
 *         |               |                     |               |
 *      +=====================+               +=====================+
 *      |         Sum         |      or       |         Sum         |
 *      +=====================+               +=====================+
 *                 |                                     |
 *         +===============+                            ***
 *         |     Relu      |
 *         +===============+
 *                 |
 *                ***
 *
 *  After:
 *
 *        ***             ***
 *         |               |
 *    +========+       +========+
 *    |  any   |-------|        |
 *    +========+       | conv2  |
 *                     |   +    |
 *                     |  sum   |
 *                     |   +    |
 *                     | [relu] |
 *                     |        |
 *                     +========+
 *                         |
 *                 +-------+
 *                 |
 *                ***
 */

void MKLDNNGraphOptimizer::FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph) {
    std::vector<MKLDNNNodePtr> &graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](MKLDNNNodePtr conv, MKLDNNNodePtr child) {
        return child->getType() == Eltwise &&
                one_of(child->getAlgorithm(), EltwiseRelu, EltwiseElu, EltwiseSigmoid, EltwiseBoundedRelu, EltwiseClamp, EltwiseSwish, EltwiseHswish,
                                              EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven, EltwiseRoundHalfAwayFromZero);
    };

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Eltwise)
            continue;

        if (graphNode->getAlgorithm() != EltwiseAdd) continue;
        if (std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isWithBroadcast()) continue;

        // TODO: Enlarge to several inputs
        bool isSutableNode = graphNode->getParentEdges().size() == 2;
        if (!isSutableNode)
            continue;

        auto parent1 = graphNode->getParentEdgeAt(0)->getParent();
        auto parent2 = graphNode->getParentEdgeAt(1)->getParent();

        bool isSutableParent1 = parent1->getType() == Convolution || parent1->getType() == BinaryConvolution;
        bool isSutableParent2 = parent2->getType() == Convolution || parent2->getType() == BinaryConvolution;

        auto* binConvNode1 = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parent1.get());
        if (binConvNode1) {
            isSutableParent1 = isSutableParent1 && binConvNode1->canFuse(graphNode);
        }

        auto* binConvNode2 = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parent2.get());
        if (binConvNode2) {
            isSutableParent2 = isSutableParent2 && binConvNode2->canFuse(graphNode);
        }

        auto* convNode1 = dynamic_cast<MKLDNNConvolutionNode *>(parent1.get());
        if (convNode1) {
            if (!convNode1->canBeExecutedInInt8()) {
                isSutableParent1 = isSutableParent1 && convNode1->getFusedWith().empty();
            }
        }

        auto* convNode2 = dynamic_cast<MKLDNNConvolutionNode *>(parent2.get());
        if (convNode2) {
            if (!convNode2->canBeExecutedInInt8()) {
                isSutableParent2 = isSutableParent2 && convNode2->getFusedWith().empty();
            }
        }

        if (!isSutableParent1 && !isSutableParent2)
            continue;

        auto mergedConv = isSutableParent1 ? parent1 : parent2;
        auto peerNode = isSutableParent1 ? parent2 : parent1;
        if (isSutableParent1 && isSutableParent2) {
            if ((peerNode->getType() == Convolution || peerNode->getType() == BinaryConvolution) &&
                mergedConv->getChildEdges().size() != 1) {
                mergedConv = parent2;
                peerNode = parent1;
            }
        }
        if (peerNode->isConstant())
            continue;
        auto sum = graphNode;

        if (mergedConv->isConstant() && !sum->isConstant())
            continue;

        auto lastNode = sum;

        bool fuse_allowed = mergedConv->getChildEdges().size() == 1;
        for (size_t j = 0; fuse_allowed && j < mergedConv->getParentEdges().size(); j++)
            if (mergedConv->getParentEdgeAt(j)->getParent() == peerNode)
                fuse_allowed = false;

        // Fused Conv+Sum prim will be used inplace. That's mean that input blob will
        // be overwritten. Should verify that all other consumer already read it and
        // we can spoil input data.
        // TODO: rewrite once we add "Inplace" reporting mechanism
        for (auto & edge : peerNode->getChildEdges()) {
            if (!fuse_allowed)
                break;
            fuse_allowed &= is_data_dependency(edge.lock()->getChild(), sum);
        }
        if (!fuse_allowed) continue;

        if (graphNode->getChildEdges().size() == 1 &&
                isFusingSupported(graphNode, graphNode->getChildEdgeAt(0)->getChild())) {
            auto relu_shared = graphNode->getChildEdgeAt(0)->getChild();
            lastNode = relu_shared;
            if (mergedConv->isConstant() && !lastNode->isConstant())
                continue;
            sum->fuseInto(mergedConv);
        }

        lastNode->fuseInto(mergedConv);

        if (mergedConv->fusedWith.size() > 0 &&
           (mergedConv->fusedWith[0]->getType() == Convolution || mergedConv->fusedWith[0]->getType() == BinaryConvolution)) {
            // Merged with DW_conv. Shape may change
            mergedConv->inDims.push_back(mergedConv->fusedWith[0]->outDims[0]);
        } else {
            mergedConv->inDims.push_back(mergedConv->outDims[0]);
        }

        size_t childIdx = 0lu;
        for (; childIdx < peerNode->getChildEdges().size(); childIdx++) {
            if (peerNode->getChildEdgeAt(childIdx)->getChild() == sum) {
                break;
            }
        }

        int peer_port = peerNode->getChildEdgeAt(childIdx)->getInputNum();
        peerNode->getChildEdgeAt(childIdx)->drop();

        int childPort = 1;
        auto* mergedConvNode = dynamic_cast<MKLDNNConvolutionNode*>(mergedConv.get());
        if (mergedConvNode != nullptr)
            childPort = mergedConvNode->getParentEdges().size();

        auto* mergedBinConvNode = dynamic_cast<MKLDNNBinaryConvolutionNode*>(mergedConv.get());
        if (mergedBinConvNode != nullptr)
            childPort = mergedBinConvNode->getParentEdges().size();

        MKLDNNEdgePtr edgePtr(new MKLDNNEdge(peerNode, mergedConv, peer_port, childPort));
        graph.GetEdges().push_back(edgePtr);

        mergedConv->addEdge(edgePtr);

        std::vector<MKLDNNEdgeWeakPtr> edges_to_reconnect = lastNode->getChildEdges();
        for (auto &edge_w : edges_to_reconnect) {
            auto edge = edge_w.lock();
            auto child = edge->getChild();
            int idxParent = edge->getInputNum();
            int idxChild = edge->getOutputNum();

            // reconnect after  activation/sum. Port index must be 0
            IE_ASSERT(idxParent == 0);

            edge->drop();

            MKLDNNEdgePtr newEdge(new MKLDNNEdge(mergedConv, child, idxParent, idxChild));
            graph.GetEdges().push_back(newEdge);
            child->addEdge(newEdge);
        }

        if (lastNode != sum) {
            lastNode->remove();
        }
        sum->remove();
    }
}

void MKLDNNGraphOptimizer::FuseMVNAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableMVN = (node->getType() == MVN) && (node->inDims[0].ndims() == 4 || node->inDims[0].ndims() == 5);

        if (isSutableMVN) {
            auto mvnNode = std::dynamic_pointer_cast<MKLDNNMVNNode>(node);
            if (mvnNode == nullptr)
                IE_THROW() << "CPU node with name '" << node->getName() << "' is not a MVN node.";

            return mvnNode->getChildEdges().size() == 1 && !mvnNode->getAcrossChannels() && mvnNode->getNormalizeVariance();
        } else {
            return false;
        }
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == MVN)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseInterpolateAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Interpolate && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        // Avoid cycle dependencies
        for (auto &childParentEdge : childNode->getParentEdges()) {
            for (auto &parentParentEdge : parentNode->getParentEdges()) {
                if (childParentEdge.lock()->getParent() == parentParentEdge.lock()->getParent())
                    return false;
            }
        }
        if (!childNode->getFusedWith().empty())
            return false;
        auto interpolateNode = dynamic_cast<MKLDNNInterpolateNode*>(parentNode.get());
        return interpolateNode->canFuse(childNode);
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Interpolate)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseNormalizeL2AndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == NormalizeL2 && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == NormalizeL2)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseEltwiseAndSimple(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Eltwise && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (parentNode->isConstant() && !childNode->isConstant())
            return false;
        for (auto &childParentEdge : childNode->getParentEdges()) {
            // WA to prevent unsupported reorder exception issue in some cases
            if (childParentEdge.lock()->getParent()->getType() == Split) {
                return false;
            }

            // Avoid cycle dependencies
            for (auto &parentParentEdge : parentNode->getParentEdges()) {
                if (childParentEdge.lock()->getParent() == parentParentEdge.lock()->getParent())
                    return false;
            }
        }

        if (!childNode->getFusedWith().empty())
            return false;

        auto eltwiseNode = dynamic_cast<MKLDNNEltwiseNode*>(parentNode.get());
        return eltwiseNode->canFuse(childNode);
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Eltwise)
                    continue;

                removeEdge(graph, p_edge);
            }

            graph.DropNode(childNode);
        } else if (childNode->getType() == Eltwise) {
            auto childs = childNode->childEdges;
            auto parents = childNode->parentEdges;

            for (size_t i = 0; i < parents.size(); i++) {
                auto p_edge = parents[i].lock();
                if (!p_edge) continue;
                auto parent = p_edge->getParent();
                if (!parent) continue;

                if (parent == parentNode) {
                    for (size_t j = 0; j < childs.size(); j++) {
                        if (!childs[j].lock())
                            continue;
                        auto child = childs[j].lock()->getChild();
                        if (!child)
                            continue;

                        MKLDNNEdgePtr &remEdge = p_edge;
                        int inNum = 0;
                        if (remEdge) {
                            inNum = remEdge->getInputNum();
                            remEdge->drop();
                            removeEdge(graph, remEdge);
                        }
                        remEdge = childs[j].lock();
                        int outNum = 0;
                        if (remEdge) {
                            outNum = remEdge->getOutputNum();
                            remEdge->drop();
                            removeEdge(graph, remEdge);
                        }
                        MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
                        auto &graphEdges = graph.GetEdges();
                        graphEdges.push_back(newEdge);
                        parent->addEdge(newEdge);

                        parent->outDims[inNum] = child->inDims[outNum];
                    }
                } else {
                    MKLDNNEdgePtr &remEdge = p_edge;
                    int inNum = 0;
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        remEdge->drop();
                        removeEdge(graph, remEdge);
                    }

                    auto parentEltwise = parentNode;
                    MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);

                    parentEltwise->inDims.push_back(parent->outDims[0]);
                }
            }

            graph.DropNode(childNode);
        } else {
            graph.DropNode(childNode);
        }
    }
}

void MKLDNNGraphOptimizer::DropDoubleReorders(MKLDNNGraph &graph) {
    std::set<MKLDNNNodePtr> processed;
    int graphNodesSize = graph.GetNodes().size();
    for (int i = 0; i < graphNodesSize; i++) {
        MKLDNNNodePtr& node = graph.GetNodes()[i];
        if (processed.find(node) == processed.end() && node->getType() == Reorder
            && node->getChildEdges().size() == 1
            && node->getChildEdgeAt(0)->getChild()->getType() == Reorder ) {
            auto nextNode = node->getChildEdgeAt(0)->getChild();
            MKLDNNReorderNode* n = dynamic_cast<MKLDNNReorderNode*>(node.get());
            if (n == nullptr)
                IE_THROW() << "Cannot get reorder layer " << node->getName();
            MKLDNNReorderNode* nn = dynamic_cast<MKLDNNReorderNode*>(nextNode.get());
            if (nn == nullptr)
                IE_THROW() << "Cannot get reorder layer " << nextNode->getName();

            auto scales = n->_scales;

            if (n->_scales != nullptr && nn->_scales != nullptr) {
                IE_THROW() << "Merging scales of two subsequent reorders is unsupported yet";
            } else {
                if (scales == nullptr) {
                    scales = nn->_scales;
                }
            }

            MKLDNNNodePtr p = n->getParentEdgeAt(0)->getParent();
            MKLDNNNodePtr c = nn->getChildEdgeAt(0)->getChild();

            auto oldEdgeNum = n->getParentEdgeAt(0)->getInputNum();

            graph.DropNode(node);
            graph.DropNode(nextNode);

            processed.insert(node);
            processed.insert(nextNode);

            MKLDNNEdgePtr edge;
            for (auto cur : p->getChildEdgesAtPort(oldEdgeNum)) {
                if (cur->getChild() == c)
                    edge = cur;
            }
            if (!edge) IE_THROW() << "Inappropriate graph processing";


            std::string layerName = edge->getParent()->getName() + "_ScaleReorder_" + edge->getChild()->getName();
            graph.InsertReorder(edge, layerName, n->getInput(), nn->getOutput(), false, scales);
            graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), edge), graph.GetEdges().end());
        }
    }
}

// TODO [NM]: reuse common/general_utils version
bool MKLDNNGraphOptimizer::IsOneOf(Type type, std::vector<Type> types) {
    for (auto tp : types) {
        if (type == tp) {
            return true;
        }
    }
    return false;
}

void MKLDNNGraphOptimizer::removeEdge(MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
    auto& edges = graph.GetEdges();
    for (auto it = edges.begin(); it != edges.end(); it++) {
        if ((*it) == edge) {
            edges.erase(it);
            return;
        }
    }
}

void MKLDNNGraphOptimizer::FuseBroadcastAndEltwise(MKLDNNGraph &graph) {
    std::vector<MKLDNNNodePtr>& graphNodes = graph.GetNodes();

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Generic
                || graphNode->getTypeStr() != "Broadcast"
                || graphNode->getChildEdges().size() != 1lu
                || graphNode->getChildEdgeAt(0)->getChild()->getType() != Eltwise)
            continue;

        MKLDNNNodePtr& broadcastNode = graphNode;
        MKLDNNNodePtr eltwiseNode = broadcastNode->getChildEdgeAt(0)->getChild();
        eltwiseNode->inDims[broadcastNode->getChildEdgeAt(0)->getOutputNum()]
                = broadcastNode->getParentEdgeAt(0)->getDims();

        auto& edges = graph.GetEdges();
        for (size_t i = 1lu; i < broadcastNode->getParentEdges().size(); i++) {
            auto constParent = broadcastNode->getParentEdgeAt(i)->getParent();
            for (auto it = edges.begin(); it != edges.end(); it++) {
                if ((*it) == constParent->getChildEdgeAt(0)) {
                    edges.erase(it);
                    constParent->remove();
                    break;
                }
            }
        }
        graph.DropNode(broadcastNode);
    }
}

void MKLDNNGraphOptimizer::FuseClampAndFakeQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableClampNode = [](MKLDNNNodePtr node) {
        return node->getType() == Eltwise && node->getChildEdges().size() == 1 && node->getAlgorithm() == EltwiseClamp;
    };

    auto isSutableFakeQuantizeNode = [](MKLDNNNodePtr node) {
        return node->getType() == FakeQuantize && node->getAlgorithm() != FQBinarization;
    };

    auto fuseClampAndFakeQuantizeNodes = [](MKLDNNNodePtr parent, MKLDNNNodePtr child) {
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent.get());
        if (eltwiseNode == nullptr)
            IE_THROW() << "Cannot cast " << parent->getName() << " to Eltwise node";

        auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode*>(child.get());
        if (fakeQuantizeNode == nullptr)
            IE_THROW() << "Cannot cast " << child->getName() << " to FakeQuantize node";

        const std::vector<float>& cropLowData = fakeQuantizeNode->getCropLow();
        const std::vector<float>& cropHighData = fakeQuantizeNode->getCropHigh();

        std::vector<float> newCropLow(cropLowData.size());
        std::vector<float> newCropHigh(cropHighData.size());
        for (int i = 0; i < cropLowData.size(); i++)
            newCropLow[i] = std::max(cropLowData[i], eltwiseNode->getAlpha());
        for (int i = 0; i < cropHighData.size(); i++)
            newCropHigh[i] = std::min(cropHighData[i], eltwiseNode->getBeta());

        fakeQuantizeNode->setCropLow(newCropLow);
        fakeQuantizeNode->setCropHigh(newCropHigh);

        return true;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableClampNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableFakeQuantizeNode(child)) continue;

        if (fuseClampAndFakeQuantizeNodes(parent, child)) {
            graph.DropNode(parent);
        }
    }
}

void MKLDNNGraphOptimizer::FuseMulAddAndFakeQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableScaleShiftNode = [](MKLDNNNodePtr node) {
        return node->getType() == Eltwise && node->getChildEdges().size() == 1 && node->getAlgorithm() == EltwiseMulAdd && node->canBePerformedAsScaleShift();
    };

    auto isSutableFakeQuantizeNode = [](MKLDNNNodePtr node) {
        return node->getType() == FakeQuantize && node->getAlgorithm() != FQBinarization;
    };

    auto fuseScaleShiftAndFakeQuantizeNodes = [](MKLDNNNodePtr parent, MKLDNNNodePtr child) {
        auto fakeQuantizeNode = std::dynamic_pointer_cast<MKLDNNFakeQuantizeNode>(child);
        if (fakeQuantizeNode == nullptr)
            IE_THROW() << "Cannot cast " << child->getName() << " to FakeQuantize node";

        auto scalesBlob = std::dynamic_pointer_cast<MKLDNNInputNode>(parent->getParentEdgesAtPort(1)[0]->getParent())->getConstBlob();
        auto shiftsBlob = std::dynamic_pointer_cast<MKLDNNInputNode>(parent->getParentEdgesAtPort(2)[0]->getParent())->getConstBlob();

        if (scalesBlob->size() != shiftsBlob->size())
            return false;

        std::vector<float> scalesBuffer;
        const float* scalesBufferPtr = scalesBlob->cbuffer().as<float*>();
        std::vector<float> shiftsBuffer;
        const float* shiftsBufferPtr = shiftsBlob->cbuffer().as<float*>();

        if (scalesBlob->getTensorDesc().getPrecision() != Precision::FP32) {
            scalesBuffer.resize(scalesBlob->size());
            cpu_convert(scalesBufferPtr, &scalesBuffer[0], scalesBlob->getTensorDesc().getPrecision(), Precision::FP32, scalesBlob->size());
            scalesBufferPtr = &scalesBuffer[0];
        }
        for (int i = 0; i < scalesBlob->size(); i++)
            if (scalesBufferPtr[i] <= 0.f)
                return false;

        if (shiftsBlob->getTensorDesc().getPrecision() != Precision::FP32) {
            shiftsBuffer.resize(shiftsBlob->size());
            cpu_convert(shiftsBufferPtr, &shiftsBuffer[0], shiftsBlob->getTensorDesc().getPrecision(), Precision::FP32, shiftsBlob->size());
            shiftsBufferPtr = &shiftsBuffer[0];
        }

        const std::vector<float>& cropLowData = fakeQuantizeNode->getCropLow();
        const std::vector<float>& cropHighData = fakeQuantizeNode->getCropHigh();
        const std::vector<float>& inputScaleData = fakeQuantizeNode->getInputScale();
        const std::vector<float>& inputShiftData = fakeQuantizeNode->getInputShift();

        std::vector<float> newCropLow(scalesBlob->size());
        std::vector<float> newCropHigh(scalesBlob->size());
        std::vector<float> newInputScale(scalesBlob->size());
        std::vector<float> newInputShift(scalesBlob->size());

        for (int i = 0; i < newCropLow.size(); i++) {
            float cl = cropLowData.size() == 1 ? cropLowData[0] : cropLowData[i];

            newCropLow[i] = (cl - shiftsBufferPtr[i]) / scalesBufferPtr[i];
        }

        for (int i = 0; i < newCropHigh.size(); i++) {
            float ch = cropHighData.size() == 1 ? cropHighData[0] : cropHighData[i];

            newCropHigh[i] = (ch - shiftsBufferPtr[i]) / scalesBufferPtr[i];
        }

        for (int i = 0; i < newInputScale.size(); i++) {
            float isc = inputScaleData.size() == 1 ? inputScaleData[0] : inputScaleData[i];

            newInputScale[i] = isc * scalesBufferPtr[i];
        }

        for (int i = 0; i < newInputShift.size(); i++) {
            float isc = inputScaleData.size() == 1 ? inputScaleData[0] : inputScaleData[i];
            float ish = inputShiftData.size() == 1 ? inputShiftData[0] : inputShiftData[i];

            newInputShift[i] = ish + shiftsBufferPtr[i] * isc;
        }

        fakeQuantizeNode->setCropLow(newCropLow);
        fakeQuantizeNode->setCropHigh(newCropHigh);
        fakeQuantizeNode->setInputScale(newInputScale);
        fakeQuantizeNode->setInputShift(newInputShift);

        return true;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableScaleShiftNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableFakeQuantizeNode(child)) continue;

        if (fuseScaleShiftAndFakeQuantizeNodes(parent, child)) {
            auto parentEdges = parent->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (!p_edge->getParent()->isConstant())
                    continue;

                removeEdge(graph, p_edge);
            }

            graph.DropNode(parent);
        }
    }
}

void MKLDNNGraphOptimizer::MergeTransposeAndReorder(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Transpose && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        return node->getType() == Reorder && node->getChildEdges().size() == 1;
    };

    // Method checkAscendingSummaryOrder() checks that after the sequential execution of Transpose and Reorder nodes,
    // the order of the elements in the memory will not change. In other words, that Transpose+Reorder is identical permutation.
    auto checkAscendingSummaryOrder = [](std::shared_ptr<MKLDNNNode> &parentNode, std::shared_ptr<MKLDNNNode> &childNode) -> bool {
        auto* transposeNode = dynamic_cast<MKLDNNTransposeNode*>(parentNode.get());
        auto* reorderNode = dynamic_cast<MKLDNNReorderNode*>(childNode.get());
        if (!transposeNode || !reorderNode) {
            return false;
        }

        auto& transposeOrder = transposeNode->getOrder();
        auto& layoutOrder = transposeNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc.getBlockingDesc().getOrder();
        auto& inOrder = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getBlockingDesc().getOrder();
        auto& outOrder = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc.getBlockingDesc().getOrder();

        if (transposeOrder.size() != layoutOrder.size() || layoutOrder.size() != inOrder.size() || inOrder.size() != outOrder.size()) {
            return false;
        }

        // revLayoutOrder - reverse permutation for layoutOrder
        auto revLayoutOrder = SizeVector(layoutOrder.size());
        for (int i = 0; i < revLayoutOrder.size(); i++) {
            revLayoutOrder[layoutOrder[i]] = i;
        }

        // newTransposeOrder - Transpose layout-aware permutation
        auto newTransposeOrder = SizeVector(transposeOrder.size());
        for (int i = 0; i < newTransposeOrder.size(); i++) {
            newTransposeOrder[i] = layoutOrder[transposeOrder[revLayoutOrder[i]]];
        }

        // reorderOrder - Reorder layout-aware permutation
        auto reorderOrder = SizeVector(outOrder.size());
        for (int i = 0; i < reorderOrder.size(); i++) {
            for (int j = 0; j < reorderOrder.size(); j++) {
                if (outOrder[i] == inOrder[j]) {
                    reorderOrder[i] = j;
                    continue;
                }
            }
        }

        // summaryOrder - resulting Transpose+Reorder permutation
        auto summaryOrder = SizeVector(transposeOrder.size());
        for (int i = 0; i < summaryOrder.size(); i++) {
            summaryOrder[i] = reorderOrder[newTransposeOrder[i]];
        }

        // check that Transpose+Reorder is the identical permutation
        for (int i = 0; i < summaryOrder.size(); i++) {
            if (summaryOrder[i] != i) {
                return false;
            }
        }

        return true;
    };

    // Transpose and Reorder do opposite permutation to each other.
    // Example:
    //      chain [physical layout: NCHW, logical layout: NCHW] -> Transpose(order=0312) -> [physical layout: NWCH, logical layout: NCHW] ->
    //      Reorder(nchw->nhwc) -> [physical layout: NCHW, logical layout: NHWC] can be replaced with Reorder(nchw->nhwc; isOptimized=true)
    //      which will just reinterprets layout without physical change of the memory.
    // Two cases are possible:
    //      1) inPrec = outPrec
    //          In this case, we replace Transpose+Reorder pattern with a new Reorder that does nothing.
    //      2) inPrec != outPrec
    //          As in the first case, we also replace Transpose+Reorder pattern with a new Reorder.
    //          Additionally, we insert another Reorder that performs the conversion from the input precision (inPrec)
    //          to the output precision (outPrec)
    auto mergeTransposeAndReorder = [&](std::shared_ptr<MKLDNNNode>& parentNode, std::shared_ptr<MKLDNNNode>& childNode) {
        auto parentParentNode = parentNode->getParentEdgesAtPort(0)[0]->getParent();
        auto parentParentConstNode = parentNode->getParentEdgesAtPort(1)[0]->getParent();
        auto childChildNode = childNode->getChildEdgeAt(0)->getChild();

        auto &remEdge = parentParentConstNode->getChildEdgeAt(0);
        remEdge->drop();
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == remEdge) {
                edges.erase(it);
                parentParentConstNode->remove();
                break;
            }
        }

        graph.DropNode(parentNode);
        graph.DropNode(childNode);

        auto inDesc = parentNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc;
        auto outDesc = childNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc;

        auto inPrec = inDesc.getPrecision();
        auto outPrec = outDesc.getPrecision();

        auto reorderInDesc = TensorDesc(inDesc);
        auto reorderOutDesc = TensorDesc(outDesc);
        reorderOutDesc.setPrecision(inPrec);

        std::string reorderlayerName = parentParentNode->getName() + "_" +
                MKLDNNExtensionUtils::getReorderArgs(reorderInDesc, reorderOutDesc) + "_" + "fake";

        MKLDNNEdgePtr edge;
        for (auto &childEdge : parentParentNode->getChildEdges()) {
            if (childEdge.lock()->getChild() == childChildNode) {
                edge = childEdge.lock();
                break;
            }
        }
        if (!edge) {
            IE_THROW() << "Transpose node '" << parentNode->getName() << "' has invalid edges.";
        }

        auto reorderNode = graph.InsertReorder(edge, reorderlayerName, reorderInDesc, reorderOutDesc, true);

        // case 2
        if (inPrec != outPrec) {
            auto reorderInDesc2 = TensorDesc(reorderOutDesc);
            auto reorderOutDesc2 = TensorDesc(outDesc);

            std::string reorderLayerName2 = reorderNode->getName() + "_" +
                                    MKLDNNExtensionUtils::getReorderArgs(reorderInDesc2, reorderOutDesc2) + "_" + childChildNode->getName();

            graph.InsertReorder(reorderNode->getChildEdgeAt(0), reorderLayerName2, reorderInDesc2, reorderOutDesc2, false);
        }
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSutableParentNode(parentNode)) {
            continue;
        }
        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(childNode)) {
            continue;
        }

        if (checkAscendingSummaryOrder(parentNode, childNode)) {
            mergeTransposeAndReorder(parentNode, childNode);
        }
    }
}
