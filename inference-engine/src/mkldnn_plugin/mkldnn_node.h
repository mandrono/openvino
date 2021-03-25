// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <memory>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <caseless.hpp>
#include <ie_common.h>
#include "mkldnn_dims.h"
#include "mkldnn_memory.h"
#include "mkldnn_edge.h"
#include "mkldnn_descriptor.h"
#include "mkldnn_selective_build.h"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_primitive.h"
#include "mkldnn_weights_cache.hpp"
#include "mkldnn.hpp"
#include <openvino/itt.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include "utils/ngraph_utils.hpp"
#include <ngraph/ops.hpp>
#include <ngraph/node.hpp>
#include <ie_precision.hpp>

namespace MKLDNNPlugin {

using MKLDNNNodePtr = std::shared_ptr<MKLDNNNode>;
using MKLDNNNodeWeakPtr = std::weak_ptr<MKLDNNNode>;

// TODO [NM]: move into separate header
enum Type {
    Unknown,
    Generic,
    Reorder,
    Input,
    Output,
    Convolution,
    Deconvolution,
    Activation,
    Depthwise,
    Lrn,
    Pooling,
    FullyConnected,
    Softmax,
    Split,
    Concatenation,
    Eltwise,
    Gemm,
    Crop,
    Reshape,
    Tile,
    SimplerNMS,
    ROIAlign,
    ROIPooling,
    BatchNormalization,
    Flatten,
    Pad,
    Transpose,
    Copy,
    MemoryOutput,
    MemoryInput,
    RNNCell,
    RNNSeq,
    Quantize,
    BinaryConvolution,
    DeformableConvolution,
    TensorIterator,
    Convert,
    MVN,
    NormalizeL2,
    ScatterUpdate,
    ScatterElementsUpdate,
    ScatterNDUpdate,
    Interpolate,
    ReduceAnd,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceOr,
    ReduceProd,
    ReduceSum,
    ReduceSumSquare,
    Reference
};

enum Algorithm {
    Undefined,

    // Pooling algorithms
    PoolingMax,
    PoolingAvg,

    // Convolution algorithms
    ConvolutionCommon,
    ConvolutionGrouped,

    // Convolution algorithms
    DeconvolutionCommon,
    DeconvolutionGrouped,

    // Elementwise algorithms
    EltwiseAdd,
    EltwiseMultiply,
    EltwiseSubtract,
    EltwiseDivide,
    EltwiseFloorMod,
    EltwiseMod,
    EltwiseMaximum,
    EltwiseMinimum,
    EltwiseSquaredDifference,
    EltwisePowerDynamic,
    EltwisePowerStatic,
    EltwiseMulAdd,
    EltwiseEqual,
    EltwiseNotEqual,
    EltwiseGreater,
    EltwiseGreaterEqual,
    EltwiseLess,
    EltwiseLessEqual,
    EltwiseLogicalAnd,
    EltwiseLogicalOr,
    EltwiseLogicalXor,
    EltwiseLogicalNot,
    EltwiseRelu,
    EltwiseGelu,
    EltwiseElu,
    EltwiseTanh,
    EltwiseSigmoid,
    EltwiseSquare, // TODO [NM]: looks like unused - remove
    EltwiseAbs,
    EltwiseSqrt,
    EltwiseLinear, // TODO [NM]: looks like unused - remove
    EltwiseBoundedRelu, // TODO [NM]: looks like unused - remove
    EltwiseSoftRelu, // TODO [NM]: looks like unused - remove
    EltwiseRelu6, // TODO [NM]: looks like unused - remove
    EltwiseExp,
    EltwiseClamp,
    EltwiseSwish,
    EltwisePrelu,
    EltwiseMish,
    EltwiseHswish,
    EltwiseHsigmoid,
    EltwiseRoundHalfToEven,
    EltwiseRoundHalfAwayFromZero
};

Type TypeFromName(const std::string type);

static std::string NameFromType(Type type) {
    switch (type) {
        case Generic:
            return "Generic";
        case Reorder:
            return "Reorder";
        case Input:
            return "Input";
        case Output:
            return "Output";
        case Convolution:
            return "Convolution";
        case Deconvolution:
            return "Deconvolution";
        case Activation:
            return "Activation";
        case Lrn:
            return "Lrn";
        case Pooling:
            return "Pooling";
        case FullyConnected:
            return "FullyConnected";
        case Gemm:
            return "Gemm";
        case Softmax:
            return "Softmax";
        case Split:
            return "Split";
        case Concatenation:
            return "Concatenation";
        case Depthwise:
            return "Depthwise";
        case Crop:
            return "Crop";
        case Reshape:
            return "Reshape";
        case Tile:
            return "Tile";
        case SimplerNMS:
            return "SimplerNMS";
        case ROIAlign:
            return "ROIAlign";
        case ROIPooling:
            return "ROIPooling";
        case BatchNormalization:
            return "BatchNormalization";
        case Flatten:
            return "Flatten";
        case Pad:
            return "Pad";
        case Transpose:
            return "Transpose";
        case Copy:
            return "Copy";
        case MemoryOutput:
            return "MemoryOutput";
        case MemoryInput:
            return "MemoryInput";
        case RNNSeq:
            return "RNNSeq";
        case RNNCell:
            return "RNNCell";
        case Eltwise:
            return "Eltwise";
        case Quantize:
            return "Quantize";
        case BinaryConvolution:
            return "BinaryConvolution";
        case DeformableConvolution:
            return "DeformableConvolution";
        case MVN:
            return "MVN";
        case TensorIterator:
            return "TensorIterator";
        case Convert:
            return "Convert";
        case NormalizeL2:
            return "NormalizeL2";
        case ScatterUpdate:
            return "ScatterUpdate";
        case ScatterElementsUpdate:
            return "ScatterElementsUpdate";
        case ScatterNDUpdate:
            return "ScatterNDUpdate";
        case Interpolate:
            return "Interpolate";
        case ReduceAnd:
            return "ReduceAnd";
        case ReduceL1:
            return "ReduceL1";
        case ReduceL2:
            return "ReduceL2";
        case ReduceLogSum:
            return "ReduceLogSum";
        case ReduceLogSumExp:
            return "ReduceLogSumExp";
        case ReduceMax:
            return "ReduceMax";
        case ReduceMean:
            return "ReduceMean";
        case ReduceMin:
            return "ReduceMin";
        case ReduceOr:
            return "ReduceOr";
        case ReduceProd:
            return "ReduceProd";
        case ReduceSum:
            return "ReduceSum";
        case ReduceSumSquare:
            return "ReduceSumSquare";
        default:
            return "Unknown";
    }
}

class PrimitiveDescInfo {
public:
    PrimitiveDescInfo(const InferenceEngine::LayerConfig& conf, impl_desc_type type): config(conf) {
        implementationType = type;
    }

    PrimitiveDescInfo(const InferenceEngine::LayerConfig& conf, impl_desc_type type, const std::vector<mkldnn::memory::format_tag>& outFmts): config(conf) {
        implementationType = type;
        outputLayouts = outFmts;
    }

    PrimitiveDescInfo(const InferenceEngine::LayerConfig& conf, impl_desc_type type, mkldnn::memory::format_tag outFmt): config(conf) {
        implementationType = type;

        setOutputLayouts(outFmt);
    }

    PrimitiveDescInfo(const PrimitiveDescInfo &descInfo) = default;
    PrimitiveDescInfo(PrimitiveDescInfo &&descInfo) = default;

    PrimitiveDescInfo &operator=(const PrimitiveDescInfo &descInfo) = default;

    const InferenceEngine::LayerConfig getConfig() const {
        return config;
    }
    InferenceEngine::LayerConfig& getConfig() {
        return config;
    }

    impl_desc_type getImplementationType() const {
        return implementationType;
    }

    const std::vector<mkldnn::memory::format_tag>& getOutputLayouts() const {
        return outputLayouts;
    }

    void setImplementationType(impl_desc_type type) {
        implementationType = type;
    }

    void setOutputLayouts(mkldnn::memory::format_tag outFmt) {
        outputLayouts.clear();

        for (int i = 0; i < config.outConfs.size(); i++) {
            outputLayouts.push_back(outFmt);
        }
    }

private:
    InferenceEngine::LayerConfig config;
    impl_desc_type implementationType;
    std::vector<mkldnn::memory::format_tag> outputLayouts;
};

class MKLDNNNode : public InferenceEngine::details::no_copy {
public:
    template<typename T, int N>
    struct Tag {};

    struct PerfCounters {
        PerfCounters(std::string const& name)
            : execute(openvino::itt::handle(name))
            , getSupportedDescriptors(openvino::itt::handle<Tag<MKLDNNNode, 0>>("MKLDNNNode::getSupportedDescriptors"))
            , initSupportedPrimitiveDescriptors(openvino::itt::handle<Tag<MKLDNNNode, 1>>("MKLDNNNode::initSupportedPrimitiveDescriptors"))
            , filterSupportedPrimitiveDescriptors(openvino::itt::handle<Tag<MKLDNNNode, 2>>("MKLDNNNode::filterSupportedPrimitiveDescriptors"))
            , selectOptimalPrimitiveDescriptor(openvino::itt::handle<Tag<MKLDNNNode, 3>>("MKLDNNNode::selectOptimalPrimitiveDescriptor"))
            , createPrimitive(openvino::itt::handle<Tag<MKLDNNNode, 4>>("MKLDNNNode::createPrimitive"))
            , initOptimalPrimitiveDescriptor(openvino::itt::handle<Tag<MKLDNNNode, 5>>("MKLDNNNode::initOptimalPrimitiveDescriptor"))
        {}

        template<typename NodeType>
        void buildClassCounters(const std::string& type_name) {
            getSupportedDescriptors = openvino::itt::handle<Tag<NodeType, 0>>(type_name + "::getSupportedDescriptors");
            initSupportedPrimitiveDescriptors = openvino::itt::handle<Tag<NodeType, 1>>(type_name + "::initSupportedPrimitiveDescriptors");
            filterSupportedPrimitiveDescriptors = openvino::itt::handle<Tag<NodeType, 2>>(type_name + "::filterSupportedPrimitiveDescriptors");
            selectOptimalPrimitiveDescriptor = openvino::itt::handle<Tag<NodeType, 3>>(type_name + "::selectOptimalPrimitiveDescriptor");
            createPrimitive = openvino::itt::handle<Tag<NodeType, 4>>(type_name + "::createPrimitive");
            initOptimalPrimitiveDescriptor = openvino::itt::handle<Tag<NodeType, 5>>(type_name + "::initOptimalPrimitiveDescriptor");
        }

        openvino::itt::handle_t execute;
        openvino::itt::handle_t getSupportedDescriptors;
        openvino::itt::handle_t initSupportedPrimitiveDescriptors;
        openvino::itt::handle_t filterSupportedPrimitiveDescriptors;
        openvino::itt::handle_t selectOptimalPrimitiveDescriptor;
        openvino::itt::handle_t createPrimitive;
        openvino::itt::handle_t initOptimalPrimitiveDescriptor;
    };

    class NodesFactory;
    static NodesFactory & factory();

    ~MKLDNNNode() override = default;

    void addEdge(const MKLDNNEdgeWeakPtr& edge);
    void removeEdge(const MKLDNNEdgeWeakPtr& edge);

    virtual void cleanup();
    void remove();

    const std::vector<MKLDNNEdgeWeakPtr> &getParentEdges() const noexcept {
        return parentEdges;
    }

    const std::vector<MKLDNNEdgeWeakPtr> &getChildEdges() const noexcept {
        return childEdges;
    }

    const MKLDNNEdgePtr getParentEdgeAt(size_t idx) const;
    virtual const MKLDNNEdgePtr getChildEdgeAt(size_t idx) const;

    const std::vector<MKLDNNEdgePtr> getParentEdgesAtPort(size_t idx) const;
    const std::vector<MKLDNNEdgePtr> getChildEdgesAtPort(size_t idx) const;

    bool isDropped() {
        return (isEdgesEmpty(childEdges) && isEdgesEmpty(parentEdges));
    }

    const mkldnn::engine& getEngine() const {
        return engine;
    }

    bool isConstant();

    bool isInplace() const;

    bool isFusedWith(Type type) const;

    void addFusedNode(const MKLDNNNodePtr &fusingNode) {
        fusedWith.push_back(fusingNode);
    }

    virtual void fuseInto(MKLDNNNodePtr& parentNode) {
        // The graph supports fusing only of consecutive nodes and some graph logic requires to know through which input port a node was fused into parent one.
        for (int i = 0; i < getParentEdges().size(); i++) {
            if (getParentEdgesAtPort(i)[0]->getParent().get() == parentNode.get()) {
                setFusingPort(i);
                break;
            }
        }

        auto parentFusedNodes = parentNode->getFusedWith();
        if (getFusingPort() < 0 && !parentFusedNodes.empty()) {
            for (int i = 0; i < getParentEdges().size(); i++) {
                if (getParentEdgesAtPort(i)[0]->getParent().get() == parentFusedNodes[parentFusedNodes.size() - 1].get()) {
                    setFusingPort(i);
                    break;
                }
            }
        }

        if (getFusingPort() == -1) {
            THROW_IE_EXCEPTION << "Cannot determine fusing port between nodes: " << parentNode->getName() << " and " << getName();
        }

        parentNode->addFusedNode(getParentEdgesAtPort(getFusingPort())[0]->getChild());
        parentNode->addOriginalLayer(getOriginalLayers());
    }

    void clearFusedWith() {
        fusedWith.clear();
    }

    void mergeWith(const MKLDNNNodePtr &merge) {
        mergedWith.push_back(merge);
    }

    const std::vector <MKLDNNNodePtr> &getMergeWith() {
        return mergedWith;
    }

    const std::vector <MKLDNNNodePtr> &getFusedWith() {
        return fusedWith;
    }

    int getFusingPort() const {
        return fusingPort;
    }

    void setFusingPort(int fusingPort) {
        this->fusingPort = fusingPort;
    }

    const std::string getName() const {
        return name;
    }

    void addOriginalLayer(const std::string& layerName);

    const std::string getOriginalLayers() const {
        return originalLayers;
    }

    Type getType() const {
        return type;
    }

    const std::vector<PrimitiveDescInfo>& getSupportedPrimitiveDescriptors() const {
        return supportedPrimitiveDescriptors;
    }

    inline const PrimitiveDescInfo* getSelectedPrimitiveDescriptor() const {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            selectedPrimitiveDescriptorIndex >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    inline PrimitiveDescInfo* getSelectedPrimitiveDescriptor() {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            selectedPrimitiveDescriptorIndex >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    void selectPrimitiveDescriptorByIndex(int index) {
        if (index < 0 || index >= supportedPrimitiveDescriptors.size())
            selectedPrimitiveDescriptorIndex = -1;
        else
            selectedPrimitiveDescriptorIndex = index;
    }

    std::string getPrimitiveDescriptorType();

    PerfCount &PerfCounter() { return perfCounter; }

    virtual void setDynamicBatchLim(int lim);

    void resolveNotAllocatedEdges();
    virtual void execute(mkldnn::stream strm);
    virtual void initSupportedPrimitiveDescriptors();

    /**
     * @brief Filters supportedPrimitiveDescriptors according to the input layouts specified in inputMemoryFormatsFilter
     * and output layouts specified in outputMemoryFormatsFilter
     */
    virtual void filterSupportedPrimitiveDescriptors();

    virtual void createPrimitive() = 0;

    virtual void selectOptimalPrimitiveDescriptor();
    virtual void initOptimalPrimitiveDescriptor();

    virtual void getSupportedDescriptors() = 0;
    virtual void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                                  const std::vector<InferenceEngine::TensorDesc>& outputDesc) {}
    virtual void initDescriptor(const InferenceEngine::LayerConfig& config);
    virtual bool created() const = 0;
    virtual bool created(const MKLDNNExtensionManager::Ptr& extMgr) {
        return created();
    }

    /**
     * @brief Performs Node initialization based on graph context.
     * This is an auxiliary method that allows to use information not available in Node constructor (e.g. connection information with other nodes)
     */
    virtual void init() {}

    template <class PD, class D, typename FPD = bool>
    PD createPrimitiveDescriptor(const mkldnn::primitive_attr &attr = mkldnn::primitive_attr()) {
        auto descsEqual = [](const std::vector<InferenceEngine::TensorDesc>& srcDescs,
                               const std::vector<InferenceEngine::DataConfig>& selectedDescs) {
            if (srcDescs.empty() && selectedDescs.empty())
                return true;
            if (srcDescs.empty() || selectedDescs.empty())
                return false;
            for (size_t i = 0; i < srcDescs.size() && i < selectedDescs.size(); i++) {
                if (!(srcDescs[i].getBlockingDesc() == selectedDescs[i].desc.getBlockingDesc() &&
                      srcDescs[i].getPrecision() == selectedDescs[i].desc.getPrecision() &&
                      srcDescs[i].getDims() == selectedDescs[i].desc.getDims()) &&
                      srcDescs[i].getLayout() != InferenceEngine::Layout::ANY)
                    return false;
            }
            return true;
        };

        const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr)
            IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

        for (const auto& desc : descs) {
            auto itpd = desc.createPrimitiveDescriptorIterator(engine, attr);

            while (static_cast<bool>(itpd))  {
                std::vector<InferenceEngine::TensorDesc> srcDescs;
                for (size_t i = 0; i < descInputNumbers(desc); i++)
                    srcDescs.push_back(getSrcMemDesc(itpd, i));

                std::vector<InferenceEngine::TensorDesc> dstDescs;
                for (size_t i = 0; i < descOutputNumbers(desc); i++)
                    dstDescs.push_back(getDstMemDesc(itpd, i));

                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

                if (impl_type == selected_pd->getImplementationType() &&
                    descsEqual(srcDescs, selected_pd->getConfig().inConfs) &&
                    descsEqual(dstDescs, selected_pd->getConfig().outConfs)) {
                    prepareMemory(selected_pd, itpd);
                    PD prim_desc = createPd<PD, D, FPD>(desc);
                    return {itpd.get()};
                }
                if (!itpd.next_impl())
                    break;
            }
        }

        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    int getExecIndex() const {
        return execIndex;
    }

    std::string getTypeStr() const {
        return typeStr;
    }

    virtual size_t descInputNumbers(MKLDNNDescriptor desc) {
        return desc.inputNumbers();
    }

    virtual size_t descOutputNumbers(MKLDNNDescriptor desc) {
        return desc.outputNumbers();
    }

    const PerfCounters & perfCounters() const {
        return profiling;
    }

    PerfCounters & perfCounters() {
        return profiling;
    }

    /**
     * @brief Returns runtime node precision based on input/output data types or data type used for computations
     * @return Runtime node precision
     */
    virtual InferenceEngine::Precision getRuntimePrecision() const;

    const std::vector<InferenceEngine::Precision>& getOriginalInputPrecisions() const {
        return originalInputPrecisions;
    }
    const std::vector<InferenceEngine::Precision>& getOriginalOutputPrecisions() const {
        return originalOutputPrecisions;
    }

    InferenceEngine::Precision getOriginalInputPrecisionAtPort(size_t port) const {
        if (originalInputPrecisions.size() <= port) {
            THROW_IE_EXCEPTION << "Incorrect input port number for node " << getName();
        }
        return originalInputPrecisions[port];
    }
    InferenceEngine::Precision getOriginalOutputPrecisionAtPort(size_t port) const {
        if (originalOutputPrecisions.size() <= port) {
            THROW_IE_EXCEPTION << "Incorrect output port number for node " << getName();
        }
        return originalOutputPrecisions[port];
    }

    void setOriginalInputPrecisionAtPort(size_t port, InferenceEngine::Precision precision) {
        if (originalInputPrecisions.size() <= port) {
            THROW_IE_EXCEPTION << "Incorrect input port number for node " << getName();
        }
        originalInputPrecisions[port] = precision;
    }

    void addOriginalInputPrecision(InferenceEngine::Precision precision) {
        originalInputPrecisions.push_back(precision);
    }

    size_t getOriginalInputsNumber() const {
        return originalInputPrecisions.size();
    }

    Algorithm getAlgorithm() const {
        return algorithm;
    }

    void setAlgorithm(Algorithm alg) {
        algorithm = alg;
    }

    virtual bool canFuse(const MKLDNNNodePtr& node) const {
        return false;
    }

protected:
    void setType(Type type) {
        this->type = type;
    }

    virtual int getMaxBatch();


    virtual InferenceEngine::TensorDesc getConfiguredInputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const;
    virtual InferenceEngine::TensorDesc getConfiguredOutputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const;
    virtual MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);
    virtual MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx);

    /**
     * @brief Appends new item into ops list with the information on how the node should be executed as post operation.
     * Seed node should call this routine and pass its post operations list as parameter.
     * @param ops List of fused post operations
     */
    virtual void appendPostOps(mkldnn::post_ops& ops);
    virtual std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr() const { return nullptr; }

    typedef std::function<MKLDNNMemoryDesc (mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx)>
            GetPrimitiveMemoryFormatFunc;
    std::vector<GetPrimitiveMemoryFormatFunc> internalBlobDesc;

    std::vector<MKLDNNDims> inDims;
    std::vector<MKLDNNDims> outDims;

    std::vector <MKLDNNNodePtr> fusedWith;
    std::vector <MKLDNNNodePtr> mergedWith;
    std::vector <impl_desc_type> implPriorities;
    std::vector <mkldnn::memory::format_tag> inputMemoryFormatsFilter;
    std::vector <mkldnn::memory::format_tag> outputMemoryFormatsFilter;

    std::string originalLayers;  // contains names of the original layers separated by comma

    MKLDNNNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache);
    MKLDNNNode(const std::string& type, const std::string& name, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache);

    int selectedPrimitiveDescriptorIndex = -1;
    bool permanent = false;
    bool temporary = false;
    int dynBatchLim = 0;
    enum class ConstantType {
        Unknown,
        Const,
        NoConst
    };
    ConstantType constant = ConstantType::Unknown;
    std::vector<InferenceEngine::Blob::Ptr> internalBlobs;
    std::vector<MKLDNNMemoryPtr> internalBlobMemory;
    std::vector<PrimitiveDescInfo> supportedPrimitiveDescriptors;
    std::unordered_map<int, mkldnn::memory> primArgs;
    MKLDNNPrimitive prim;
    std::vector<MKLDNNDescriptor> descs;

    InferenceEngine::Blob::Ptr ext_scales;
    MKLDNNWeightsSharing::Ptr weightCache;

    Algorithm algorithm;

    friend class MKLDNNEdge;
    friend class MKLDNNGraph;
    friend class MKLDNNGraphOptimizer;

    bool isUninitTensorDesc(const InferenceEngine::TensorDesc& desc) const;
    bool isInitConfig(const InferenceEngine::LayerConfig& config) const;
    virtual void selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority);
    virtual bool canBeInPlace() const;

    virtual const std::vector<impl_desc_type>& getPrimitivesPriority();

    virtual std::vector<mkldnn::memory::format_tag> getAvailableFormatsForDims(const MKLDNNDims& dims) const;
    int batchToProcess();

//    InferenceEngine::Blob::Ptr createInternalBlob(InferenceEngine::SizeVector dims, bool weights, bool is_grouped = false);

    InferenceEngine::Layout getWeightsLayoutByDims(InferenceEngine::SizeVector dims, bool isGrouped);

    /**
     * @brief Auxiliary function to get node input precisions
     * @return Vector of precisions based on information from node input edges. Return empty vector in case edges are not initialized yet.
     */
    virtual std::vector<InferenceEngine::Precision> getInputPrecisions() const;

    /**
     * @brief Auxiliary function to get node output precisions
     * @return Vector of precisions based on information from node output edges. Return empty vector in case edges are not initialized yet.
     */
    virtual std::vector<InferenceEngine::Precision> getOutputPrecisions() const;

private:
    std::vector<MKLDNNEdgeWeakPtr> parentEdges;
    std::vector<MKLDNNEdgeWeakPtr> childEdges;

    std::vector<InferenceEngine::Precision> originalInputPrecisions;
    std::vector<InferenceEngine::Precision> originalOutputPrecisions;

    int fusingPort;

    mkldnn::engine engine;

    std::string name;
    const std::string typeStr;
    Type type;
    int execIndex = -1;

    std::string typeToStr(Type type);

    PerfCount perfCounter;
    PerfCounters profiling;

    bool isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const;

    template <class PD, class D, typename FPD>
    typename std::enable_if<!std::is_same<FPD, bool>::value, PD>::type
    createPd(MKLDNNDescriptor desc) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        std::shared_ptr<FPD> backward_prim_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine, *backward_prim_desc_ptr);
    }

    template <class PD, class D, typename FPD>
    typename std::enable_if<std::is_same<FPD, bool>::value, PD>::type
    createPd(MKLDNNDescriptor desc) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine);
    }

    void prepareMemory(const PrimitiveDescInfo *selected_pd, mkldnn::primitive_desc_iterator& itpd);
    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2 };
    ConstantType checkConstant(LOOK look, std::vector<MKLDNNNodePtr>& checkNodes);
};

class MKLDNNNode::NodesFactory : public openvino::cc::Factory<Type,
                                            MKLDNNNode*(const std::shared_ptr<ngraph::Node>& op,
                                                        const mkldnn::engine &,
                                                        MKLDNNWeightsSharing::Ptr &)> {
public:
    NodesFactory()
        : Factory("NodesFactory") {}

    MKLDNNNode* create(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                       const MKLDNNExtensionManager::Ptr& extMgr, MKLDNNWeightsSharing::Ptr &w_cache);
};

template<typename MKLDNNNodeType>
struct MKLDNNNodeImpl : public MKLDNNNodeType {
    MKLDNNNodeImpl(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNodeType(op, eng, cache) {
        MKLDNNNodeType::perfCounters().template buildClassCounters<MKLDNNNodeType>(NameFromType(MKLDNNNodeType::getType()));
    }
};

#define REG_MKLDNN_CONCAT3_(X, Y, Z) X ## Y ## Z
#define REG_MKLDNN_CONCAT3(X, Y, Z) REG_MKLDNN_CONCAT3_(X, Y, Z)

#define REG_MKLDNN_PRIM_FOR(__prim, __type)                                                 \
static struct REG_MKLDNN_CONCAT3(Registrar4, __prim, __LINE__) {                            \
    REG_MKLDNN_CONCAT3(Registrar4, __prim, __LINE__)() {                                    \
        MKLDNNNode::factory()                                                               \
            .registerNodeIfRequired(MKLDNNPlugin, __prim, __type, MKLDNNNodeImpl<__prim>);  \
    }                                                                                       \
} REG_MKLDNN_CONCAT3(_reg_, __prim, __LINE__);

}  // namespace MKLDNNPlugin
