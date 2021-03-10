// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_input_node.h"
#include "mkldnn_extension_utils.h"
#include <string>
#include <tuple>
#include <algorithm>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>
#include <ie_ngraph_utils.hpp>
#include <blob_factory.hpp>
#include "caseless.hpp"
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace details;
using namespace ngraph::op;

MKLDNNInputNode::MKLDNNInputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    if (!one_of(op->get_type_info(), v0::Parameter::type_info, v0::Constant::type_info, v0::Result::type_info))
        THROW_IE_EXCEPTION << "CPU Input node doesn't support ngraph operation " << op->get_type_name() << " with name " << op->get_friendly_name();

    constant = ConstantType::NoConst;
    constBlob = nullptr;

    auto constOp = ngraph::as_type_ptr<ngraph::op::Constant>(op);
    if (constOp) {
        constant = ConstantType::Const;

        auto dataPrecision = convertPrecision(op->get_element_type());

        size_t shapeSize = ngraph::shape_size(op->get_shape());
        constexpr size_t byte_size{8};
        if (dataPrecision == Precision::BIN) {
            shapeSize = (shapeSize + (byte_size - 1)) / byte_size;
        }

        TensorDesc td(dataPrecision, {shapeSize}, Layout::C);

        auto blob = make_blob_with_precision(td, const_cast<void*>(constOp->get_data_ptr()));
        blob->allocate();

        constBlob = blob;
    }
}

void MKLDNNInputNode::getSupportedDescriptors() {
    if (getType() == Input) {
        if (!getParentEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
        if (getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
    } else if (getType() == Output) {
        if (getParentEdges().size() != 1)
            THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
        if (!getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
    }
}

void MKLDNNInputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    LayerConfig config;
    config.dynBatchSupport = true;
    if (getType() == Input || getType() == MemoryInput) {
        precision = getOriginalOutputPrecisionAtPort(0);
        if (precision == Precision::U16 || isMeanImage) {
            precision = Precision::FP32;
        }
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        auto mem_tdesc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
        dataConfig.desc = mem_tdesc;
        config.outConfs.push_back(dataConfig);
    } else if (getType() == Output) {
        precision = getOriginalInputPrecisionAtPort(0);
        if (precision == Precision::U16) precision = Precision::FP32;
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        auto mem_tdesc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType);
        dataConfig.desc = mem_tdesc;
        config.inConfs.push_back(dataConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNInputNode::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set for node " << getName() << ".";
}

bool MKLDNNInputNode::created() const {
    return getType() == Input || getType() == Output;
}

namespace {
    bool isDefaultOrder(const SizeVector &order) {
        return std::is_sorted(order.begin(), order.end(),
                                [](size_t a, size_t b) { return a + 1 == b; });
    }

    std::tuple<bool, size_t> isDefaultStrides(const SizeVector &strides,
                                              const SizeVector &dims) {
        if (strides.size() != dims.size())
            return std::make_tuple(false, 0);

        size_t dim = 1;

        for (size_t i = dims.size(); i-- > 0;) {
            if (strides[i] != dim)
                return std::make_tuple(false, 0);
            dim *= dims[i];
        }

        return std::make_tuple(true, dim);
    }

    bool isCompatibleTensors(const TensorDesc &lhs, const TensorDesc &rhs) {
        auto const &lhsBlockingDesc = lhs.getBlockingDesc();
        auto const &rhsBlockingDesc = rhs.getBlockingDesc();

        bool lhsDefaultStrides = false, rhsDefaultStrides = false;
        size_t lhsSize = 0lu, rhsSize = 0lu;

        std::tie(lhsDefaultStrides, lhsSize) = isDefaultStrides(lhsBlockingDesc.getStrides(), lhs.getDims());
        std::tie(rhsDefaultStrides, rhsSize) = isDefaultStrides(rhsBlockingDesc.getStrides(), rhs.getDims());

        return lhs.getPrecision() == rhs.getPrecision()
                && lhsSize == rhsSize
                && lhsDefaultStrides
                && rhsDefaultStrides
                && isDefaultOrder(lhsBlockingDesc.getOrder())
                && isDefaultOrder(rhsBlockingDesc.getOrder());
    }
}   // namespace

void MKLDNNInputNode::execute(mkldnn::stream strm) {
    if (!constBlob)
        return;
    auto dstBlob = getChildEdgeAt(0)->getBlob();

    if (constBlob->getTensorDesc() == dstBlob->getTensorDesc()
        || isCompatibleTensors(constBlob->getTensorDesc(), dstBlob->getTensorDesc())) {
        const int8_t *srcData = constBlob->cbuffer().as<int8_t *>();
        int8_t *dstData = dstBlob->buffer();

        cpu_memcpy_s(dstData, dstBlob->byteSize(), srcData, constBlob->byteSize());
    } else if (constBlob->getTensorDesc().getPrecision() == Precision::BIN ||
               dstBlob->getTensorDesc().getPrecision() == Precision::BIN) {
        size_t dstSize = dstBlob->size() / 8;
        if (constBlob->size() != dstSize) {
            THROW_IE_EXCEPTION << "Incorrect blob sizes for node " << getName();
        }

        const int8_t *srcData = constBlob->cbuffer().as<int8_t *>();
        int8_t *dstData = dstBlob->buffer();

        cpu_memcpy_s(dstData, dstSize, srcData, constBlob->byteSize());
    } else {
        if (constBlob->size() != dstBlob->size()) {
            THROW_IE_EXCEPTION << "Incorrect blob sizes for node " << getName();
        }

        switch (precision.size()) {
            case 1: {
                const int8_t *srcData = constBlob->cbuffer().as<int8_t *>();
                int8_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            case 2: {
                const int16_t *srcData = constBlob->cbuffer().as<int16_t *>();
                int16_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            case 4: {
                const int32_t *srcData = constBlob->cbuffer().as<int32_t *>();
                int32_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            case 8: {
                const int64_t *srcData = constBlob->cbuffer().as<int64_t *>();
                int64_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision for node " << getName();
        }
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Input);
REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Output);
