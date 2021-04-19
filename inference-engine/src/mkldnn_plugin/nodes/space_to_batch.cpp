// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <cassert>
#include <ngraph/opsets/opset2.hpp>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SpaceToBatchImpl: public ExtLayerBase {
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto spaceToBatch = std::dynamic_pointer_cast<const ngraph::opset2::SpaceToBatch>(op);
            if (!spaceToBatch) {
                errorMessage = "Only opset2 SpaceToBatch operation is supported";
                return false;
            }
            if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1)) == nullptr ||
                std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2)) == nullptr ||
                    std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(3)) == nullptr) {
                errorMessage = "Only constant 'block_shape', 'pads_begin', 'pads_end' are supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::string errorPrefix;

public:
    explicit SpaceToBatchImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "BatchToSpace layer with name '" + op->get_friendly_name() + "'";

            if (op->get_input_size() != 4 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input or output edges!";

            const auto precision = details::convertPrecision(op->get_input_element_type(0));
            const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
            if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
                IE_THROW() << errorPrefix << " has unsupported precision: " << precision.name();

            const SizeVector& in_dims = op->get_input_shape(0);
            const SizeVector& out_dims = op->get_output_shape(0);
            if (in_dims[1] != out_dims[1])
                IE_THROW() << errorPrefix << " has different IN and OUT channels number";

            _block_shape = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<size_t>();
            _pads_begin = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<size_t>();
            _pads_end = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(3))->cast_vector<size_t>();

            addConfig(op, {{TensorDescCreatorTypes::ncsp, precision},
                           {TensorDescCreatorTypes::ncsp},
                           {TensorDescCreatorTypes::ncsp},
                           {TensorDescCreatorTypes::ncsp}},
                          {{TensorDescCreatorTypes::ncsp, precision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: {
                process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
                break;
            }
            case 2: {
                process_data<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs);
                break;
            }
            case 4: {
                process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
                break;
            }
            case 8: {
                process_data<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs);
                break;
            }
            default: {
                if (resp) {
                    std::string errorMsg = "SpaceToBatch layer does not support precision '"
                            + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
            }
        }

        return OK;
    }

    template<typename T>
    void process_data(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept {
        const T* src_data = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& inDims = inputs[0]->getTensorDesc().getDims();
        const size_t dims_size = inDims.size();
        const auto layout = inputs[0]->getTensorDesc().getLayout();

        const int64_t IB = inDims[0];
        const int64_t IC = inDims[1];
        const int64_t ID = layout == NCDHW ? inDims[dims_size - 3] : 1lu;
        const int64_t IH = inDims[dims_size - 2];
        const int64_t IW = inDims[dims_size - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();

        const size_t OB = outDims[0];
        const size_t OC = outDims[1];
        const size_t OD = layout == NCDHW ? outDims[dims_size - 3] : 1lu;
        const size_t OH = outDims[dims_size - 2];
        const size_t OW = outDims[dims_size - 1];

        const int64_t cBSD = layout == NCDHW ? _block_shape[dims_size - 3] : 1lu;  // Do not use name BSD. It affects MacOS build
        const int64_t BSH = _block_shape[dims_size - 2];
        const int64_t BSW = _block_shape[dims_size - 1];

        const int64_t PF = layout == NCDHW ? _pads_begin[dims_size - 3] : 0;
        const int64_t PT = _pads_begin[dims_size - 2];
        const int64_t PL = _pads_begin[dims_size - 1];

        const size_t OH_OW = OH * OW;
        const size_t IH_IW = IH * IW;
        const size_t ID_IH_IW = ID * IH_IW;
        const size_t IC_ID_IH_IW = IC * ID_IH_IW;

        const size_t work_amount = OB*OC*OD*OH*OW;

        auto thread_body = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(work_amount, nthr, ithr, start, end);
            if (start >= end)
                return;
            int64_t ob(0), oc(0), od(0), oh(0), ow(0);
            parallel_it_init(start, ob, OB, oc, OC, od, OD, oh, OH, ow, OW);

            for (; ob < OB; ob++) {
                const int64_t ib = ob % IB;
                const int64_t ib_k = ib * IC_ID_IH_IW;
                int64_t bi = ob / IB;
                const int64_t shift_w = bi % BSW - PL;
                bi /= BSW;
                const int64_t shift_h = (layout == NCDHW ? bi % BSH : bi) - PT;
                const int64_t shift_d = layout == NCDHW ? (bi / BSH - PF) : 0;
                for (; oc < OC; oc++) {
                    const int64_t ic_k = ib_k + oc * ID_IH_IW;
                    for (; od < OD; od++) {
                        const int64_t id = od * cBSD + shift_d;
                        if (id < 0 || id >= ID) {
                            std::fill(dst_data + start, dst_data + start + OH_OW, T(0));
                            start += OH_OW;
                            if (start >= end)
                                break;
                            continue;
                        }
                        const int64_t id_k = ic_k + id * IH_IW;
                        for (; oh < OH; oh++) {
                            const int64_t ih = oh * BSH + shift_h;
                            if (ih < 0 || ih >= IH) {
                                std::fill(dst_data + start, dst_data + start + OW, T(0));
                                start += OW;
                                if (start >= end)
                                    break;
                                continue;
                            }
                            const int64_t ih_k = id_k + ih * IW;
                            for (; ow < OW; ow++) {
                                const int64_t iw = ow * BSW + shift_w;
                                if (iw < 0 || iw >= IW) {
                                    dst_data[start] = T(0);
                                    start++;
                                    if (start >= end)
                                        break;
                                    continue;
                                }
                                const int64_t src_idx = ih_k + iw;
                                dst_data[start] = src_data[src_idx];
                                start++;
                                if (start >= end)
                                    break;
                            }
                            if (start >= end)
                                break;
                            ow = 0;
                        }
                        if (start >= end)
                            break;
                        oh = 0;
                    }
                    if (start >= end)
                        break;
                    od = 0;
                }
                if (start >= end)
                    break;
                oc = 0;
            }
        };

        parallel_nt(0, thread_body);
    }

private:
    std::vector<size_t> _block_shape;
    std::vector<size_t> _pads_begin;
    std::vector<size_t> _pads_end;
};

REG_FACTORY_FOR(SpaceToBatchImpl, SpaceToBatch);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
