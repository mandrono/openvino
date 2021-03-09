// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_matmul_to_fc_or_gemm.hpp"
#include "fc_bias_fusion.hpp"
#include "reshape_fc_fusion.hpp"
#include "reshape_fully_connected.hpp"

namespace MKLDNNPlugin {

inline void convertToPluginSpecificNodes(ngraph::pass::Manager &passManager, const std::shared_ptr<const ngraph::Function> &nGraphFunc) {
    passManager.register_pass<ConvertMatMulToFC>();
    passManager.register_pass<ConvertMatMulToGemm>();
    passManager.register_pass<FullyConnectedBiasFusion>();
    passManager.register_pass<ReshapeFullyConnected>();
    if (!ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc)) {
        passManager.register_pass<ReshapeFullyConnectedFusion>();
    }
    passManager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
}

}  // namespace MKLDNNPlugin