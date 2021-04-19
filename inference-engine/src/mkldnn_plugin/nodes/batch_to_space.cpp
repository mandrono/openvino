// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <set>
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class BatchToSpaceImpl: public ExtLayerBase {
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto batchToSpace = std::dynamic_pointer_cast<const ngraph::opset2::BatchToSpace>(op);
            if (!batchToSpace) {
                errorMessage = "Only opset2 BatchToSpace operation is supported";
                return false;
            }
            if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1)) == nullptr ||
                std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2)) == nullptr ||
                    std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(3)) == nullptr) {
                errorMessage = "Only constant 'block_shape', 'crops_begin', 'crops_end' are supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::string errorPrefix;

public:
    explicit BatchToSpaceImpl(const std::shared_ptr<ngraph::Node>& op) {
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
            _crops_begin = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<size_t>();
            _crops_end = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(3))->cast_vector<size_t>();

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
                    std::string errorMsg = "BatchToSpace layer does not support precision '"
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

        const size_t IB = inDims[0];
        const size_t IC = inDims[1];
        const size_t ID = layout == NCDHW ? inDims[dims_size - 3] : 1lu;
        const size_t IH = inDims[dims_size - 2];
        const size_t IW = inDims[dims_size - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();

        const size_t OB = outDims[0];
        const size_t OC = outDims[1];
        const size_t OD = layout == NCDHW ? outDims[dims_size - 3] : 1lu;
        const size_t OH = outDims[dims_size - 2];
        const size_t OW = outDims[dims_size - 1];

        const int64_t cBSD = layout == NCDHW ? _block_shape[dims_size - 3] : 1lu;  // Do not use name BSD. It affects MacOS build
        const int64_t BSH = _block_shape[dims_size - 2];
        const int64_t BSW = _block_shape[dims_size - 1];

        const size_t crop_front = layout == NCDHW ? _crops_begin[dims_size - 3] : 0lu;
        const size_t crop_top = _crops_begin[dims_size - 2];
        const size_t crop_left = _crops_begin[dims_size - 1];

        const size_t OH_OW = OH * OW;
        const size_t OD_OH_OW = OD * OH_OW;
        const size_t OC_OD_OH_OW = OC * OD_OH_OW;
        const size_t IH_IW = IH * IW;

        const size_t work_amount = IB*IC*ID*IH*IW;

        auto thread_body = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(work_amount, nthr, ithr, start, end);
            if (start >= end)
                return;
            int64_t ib(0), ic(0), id(0), ih(0), iw(0);
            parallel_it_init(start, ib, IB, ic, IC, id, ID, ih, IH, iw, IW);

            for (; ib < IB; ib++) {
                const size_t ob = ib % OB;
                const size_t ob_k = ob * OC_OD_OH_OW;
                int64_t b_idx = ib / OB;
                const int64_t ow_add = b_idx % BSW - crop_left;
                b_idx /= BSW;
                const int64_t oh_add = (layout == NCDHW ? b_idx % BSH : b_idx) - crop_top;
                const int64_t od_add = layout == NCDHW ? (b_idx / BSH - crop_front) : 0;
                for (; ic < IC; ic++) {
                    const size_t oc_k = ob_k + ic * OD_OH_OW;
                    for (; id < ID; id++) {
                        const int64_t od = id * cBSD + od_add;
                        if (od < 0 || od >= OD) {
                            start += IH_IW;
                            if (start >= end)
                                break;
                            continue;
                        }
                        const size_t od_k = oc_k + od * OH_OW;
                        for (; ih < IH; ih++) {
                            const int64_t oh = ih * BSH + oh_add;
                            if (oh < 0 || oh >= OH) {
                                start += IW;
                                if (start >= end)
                                    break;
                                continue;
                            }
                            const size_t oh_k = od_k + oh * OW;
                            for (; iw < IW; iw++) {
                                const int64_t ow = iw * BSW + ow_add;
                                if (ow < 0 || ow >= OW) {
                                    start++;
                                    if (start >= end)
                                        break;
                                    continue;
                                }
                                const size_t dst_idx = oh_k + ow;
                                dst_data[dst_idx] = src_data[start];
                                start++;
                                if (start >= end)
                                    break;
                            }
                            if (start >= end)
                                break;
                            iw = 0;
                        }
                        if (start >= end)
                            break;
                        ih = 0;
                    }
                    if (start >= end)
                        break;
                    id = 0;
                }
                if (start >= end)
                    break;
                ic = 0;
            }
        };

        parallel_nt(0, thread_body);
    }

private:
    std::vector<size_t> _block_shape;
    std::vector<size_t> _crops_begin;
    std::vector<size_t> _crops_end;
};

REG_FACTORY_FOR(BatchToSpaceImpl, BatchToSpace);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
