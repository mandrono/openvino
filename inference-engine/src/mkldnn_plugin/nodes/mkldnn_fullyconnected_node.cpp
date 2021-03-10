// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_fullyconnected_node.h"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_quantize_node.h"
#include "ngraph_transformations/op/fully_connected.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <vector>
#include <mkldnn_extension_utils.h>
#include <mkldnn.hpp>
#include "utils/general_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNFullyConnectedNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto fc = std::dynamic_pointer_cast<const FullyConnectedNode>(op);
        if (!fc) {
            errorMessage = "Only legacy FullyConnected operation is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(fc->get_input_node_shared_ptr(WEIGHTS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'weights' input is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(fc->get_input_node_shared_ptr(BIAS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'bias' input is supported";
            return false;
        }
        if (fc->get_input_shape(DATA_ID).size() != 2) {
            errorMessage = "Only 'data' input with rank = 2 is supported";
            return false;
        }
        if (fc->get_input_shape(WEIGHTS_ID).size() != 2) {
            errorMessage = "Only 'weights' input with rank = 2 is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNFullyConnectedNode::MKLDNNFullyConnectedNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache), withBiases(false) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "FullyConnected node with name '" + getName() + "'";

        const auto bias = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(BIAS_ID))->cast_vector<float>();
        withBiases = std::any_of(bias.begin(), bias.end(), [](float b){ return b != 0.0f; });

    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED) << errorMessage;
    }
}

std::vector<memory::format_tag> MKLDNNFullyConnectedNode::getAvailableFormatsForDims(const MKLDNNDims &dims) const {
    if (dims.ndims() == 0)
        return {memory::format_tag::x};
    else if (dims.ndims() == 1)
        return {memory::format_tag::x};
    else if (dims.ndims() == 2)
        return {memory::format_tag::nc};
    else if (dims.ndims() == 3)
        return {memory::format_tag::tnc};
    else if (dims.ndims() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.ndims() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

void MKLDNNFullyConnectedNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 3)
        THROW_IE_EXCEPTION << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION<< errorPrefix << " has incorrect number of output edges";

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisions()[DATA_ID]);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisions()[DATA_ID]);

    if (inputDataType == memory::data_type::f32) {
        outputDataType = memory::data_type::f32;
    }

    if (!fusedWith.empty()) {
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisions()[0]);
    }
    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisions()[WEIGHTS_ID]);

    if ((!one_of(inputDataType , memory::data_type::u8, memory::data_type::s8) || weightsDataType != memory::data_type::s8)
            && inputDataType != memory::data_type::bf16) {
        inputDataType = outputDataType = memory::data_type::f32;
    }

    MKLDNNDims inDims = getParentEdgeAt(0)->getDims();
    MKLDNNDims outDims = getChildEdgeAt(0)->getDims();

    for (auto format : getAvailableFormatsForDims(inDims)) {
        MKLDNNMemoryDesc in_candidate(inDims, inputDataType, format);
        MKLDNNMemoryDesc out_candidate(outDims, outputDataType, memory::format_tag::any);

        createDescriptor({in_candidate}, {out_candidate});
    }
}

void MKLDNNFullyConnectedNode::createPrimitive() {
    if (prim)
        return;

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();
    std::shared_ptr<inner_product_forward::primitive_desc> prim_desc;
    prim_desc = std::make_shared<inner_product_forward::primitive_desc>(
            createPrimitiveDescriptor<inner_product_forward::primitive_desc, inner_product_forward::desc>(*attr));

    prim.reset(new inner_product_forward(*prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    if (withBiases)
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getParentEdgeAt(WEIGHTS_ID)->getMemory().GetPrimitive()},
                    {DNNL_ARG_BIAS, getParentEdgeAt(BIAS_ID)->getMemory().GetPrimitive()}, {DNNL_ARG_DST, dst}};
    else
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getParentEdgeAt(WEIGHTS_ID)->getMemory().GetPrimitive()}, {DNNL_ARG_DST, dst}};
}

void MKLDNNFullyConnectedNode::execute(mkldnn::stream strm) {
    if (prim) {
        // At this moment all reshape operation executed on ngraph transformation side
        //
        // auto reshapeMemory = [this](int argType) {
        //     auto param = primArgs.find(argType);
        //     if (param != primArgs.end()) {
        //         auto oldMem = param->second;
        //         auto dims = oldMem.get_desc().dims();
        //         if (dims.size() == 3) {
        //             MKLDNNDims normalizedDims({static_cast<ptrdiff_t>(dims[0] * dims[1]), static_cast<ptrdiff_t>(dims[2])});
        //             mkldnn::memory::desc newMemDesc(oldMem.get_desc().reshape(normalizedDims));
        //             mkldnn::memory newMem(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());
        //             primArgs.at(argType) = newMem;
        //         }
        //     }
        // };
        // reshapeMemory(DNNL_ARG_SRC);
        // reshapeMemory(DNNL_ARG_DST);

        (*prim).execute(strm, primArgs);
    }
}

void MKLDNNFullyConnectedNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) {
//    int blob_idx = 0;
   mkldnn::post_ops ops;
//
//    for (auto &node : fusedWith) {
//        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
//        if (quantizeNode) {
//            quantizeNode->appendPostOps(ops);
//            continue;
//        }
//
//        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
//        if (eltwiseNode && (eltwiseNode->getOpType() == MulAdd || eltwiseNode->getOpType() == Prelu)) {
//            if (initWeights) {
//                auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(eltwiseNode->getCnnLayer().get());
//                int ndims = getParentEdgeAt(0)->getDims().ndims();
//                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(ndims == 3 ? getChildEdgeAt(0)->getDims()[2] : getChildEdgeAt(0)->getDims()[1], 16))});
//
//                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
//                PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format_tag::x);
//                PostOpsIntBlobMemory[blob_idx]->FillZero();
//
//                // In case ndims == 3 graph optimizer allows fusing only if all weights values are the same
//                if (depthwiseLayer->blobs["weights"]->size() == 1 || ndims == 3) {
//                    float broadcastValue = static_cast<float *>(depthwiseLayer->_weights->buffer())[0];
//                    for (int i = 0; i < PostOpsIntBlobMemory[blob_idx]->GetDesc().getDims()[0]; i++) {
//                        static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
//                    }
//                } else {
//                    PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::format_tag::x,
//                                                            depthwiseLayer->_weights->buffer(),
//                                                            depthwiseLayer->_weights->size() *
//                                                            MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));
//                }
//
//                if (eltwiseNode->getAlgorithm() == algorithm::depthwise_scale_shift) {
//                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
//                    PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32, memory::format_tag::x);
//                    PostOpsIntBlobMemory[blob_idx + 1]->FillZero();
//
//                    // In case ndims == 3 graph optimizer allows fusing only if all biases values are the same
//                    if (depthwiseLayer->blobs["biases"]->size() == 1 || ndims == 3) {
//                        float broadcastValue = static_cast<float *>(depthwiseLayer->_biases->buffer())[0];
//                        for (int i = 0; i < PostOpsIntBlobMemory[blob_idx + 1]->GetDesc().getDims()[0]; i++) {
//                            static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
//                        }
//                    } else {
//                        PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::format_tag::x,
//                                                                    depthwiseLayer->_biases->buffer(),
//                                                                    depthwiseLayer->_biases->size() *
//                                                                    MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));
//                    }
//
//                    ops.append_depthwise(eltwiseNode->getAlgorithm(),
//                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
//                                         (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());
//
//                    blob_idx += 2;
//                } else {
//                    ops.append_depthwise(eltwiseNode->getAlgorithm(),
//                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
//                                         nullptr);
//
//                    blob_idx += 1;
//                }
//            } else {
//                ops.append_depthwise(eltwiseNode->getAlgorithm(),
//                                     nullptr,
//                                     nullptr);
//            }
//
//            continue;
//        }
//
//        if (eltwiseNode) {
//            eltwiseNode->appendPostOps(ops);
//        }
//    }
//
   attr.set_post_ops(ops);
}

bool MKLDNNFullyConnectedNode::created() const {
    return getType() == FullyConnected;
}

const std::vector<impl_desc_type>& MKLDNNFullyConnectedNode::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx,
            impl_desc_type::gemm_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm,
            impl_desc_type::jit_gemm,
            impl_desc_type::jit_uni_dw,
            impl_desc_type::jit_uni_1x1,
            impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw,
            impl_desc_type::jit_avx512_1x1,
            impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw,
            impl_desc_type::jit_avx2_1x1,
            impl_desc_type::jit_avx2,
            impl_desc_type::jit_avx_dw,
            impl_desc_type::jit_avx_1x1,
            impl_desc_type::jit_avx,
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

std::shared_ptr<mkldnn::primitive_attr> MKLDNNFullyConnectedNode::initPrimitiveAttr() {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, true);

    return attr;
}

void MKLDNNFullyConnectedNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                                const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    TensorDesc inDesc = inputDesc[0], outDesc = outputDesc[0];

    mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    mkldnn::memory::data_type bdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    if (inDesc.getPrecision() == Precision::BF16) {
        bdt = mkldnn::memory::data_type::f32;
    } else if (inDesc.getPrecision() == Precision::U8 || inDesc.getPrecision() == Precision::I8) {
        wdt = memory::data_type::s8;
        bdt = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisions()[BIAS_ID]);
    }

    MKLDNNMemoryDesc in_candidate(inDesc);
    MKLDNNMemoryDesc out_candidate(outDesc);
    MKLDNNMemoryDesc wgh_candidate(MKLDNNDims(inDims[WEIGHTS_ID]), wdt, mkldnn::memory::format_tag::any);

    if (withBiases) {
        MKLDNNMemoryDesc bias_candidate(MKLDNNDims(inDims[BIAS_ID]), bdt, memory::format_tag::any);
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                bias_candidate, out_candidate)));
        descs.push_back(desc);
    } else {
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                out_candidate)));
        descs.push_back(desc);
    }
}

MKLDNNMemoryDesc MKLDNNFullyConnectedNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    if (idx == 2 && !withBiases) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(getOriginalInputPrecisions()[BIAS_ID],
                                                            getParentEdgeAt(BIAS_ID)->getDims().ToSizeVector(),
                                                            TensorDesc::getLayoutByDims(getParentEdgeAt(BIAS_ID)->getDims().ToSizeVector())));
    }

    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
                                               : MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

    if (desc.getLayout() == InferenceEngine::Layout::ANY) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    } else {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
    }
}

MKLDNNMemoryDesc MKLDNNFullyConnectedNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.dst_desc(idx));
    if (desc.getLayout() == InferenceEngine::Layout::ANY) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    } else {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
    }
}

InferenceEngine::Precision MKLDNNFullyConnectedNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return MKLDNNExtensionUtils::getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNFullyConnectedNode, FullyConnected);
