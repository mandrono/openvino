// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include <ngraph/op/util/rnn_cell_base.hpp>

namespace MKLDNNPlugin {

class LSTMSequenceNode : public ngraph::op::util::RNNCellBase {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"LSTMSequenceCPU", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }

    LSTMSequenceNode(const ngraph::Output<ngraph::Node>& X,
                     const ngraph::Output<ngraph::Node>& H_t,
                     const ngraph::Output<ngraph::Node>& C_t,
                     const ngraph::Output<ngraph::Node>& sequence_lengths,
                     const ngraph::Output<ngraph::Node>& W,
                     const ngraph::Output<ngraph::Node>& R,
                     const ngraph::Output<ngraph::Node>& B,
                     std::size_t hidden_size,
                     ngraph::op::RecurrentSequenceDirection direction,
                     const std::vector<std::string>& activations,
                     const std::vector<float>& activations_alpha,
                     const std::vector<float>& get_activations_beta,
                     float clip,
                     int64_t seq_axis);

    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;

    ngraph::op::RecurrentSequenceDirection get_direction() const { return m_direction; }

private:
    ngraph::op::RecurrentSequenceDirection m_direction;
    int64_t m_seq_axis;
};

}  // namespace MKLDNNPlugin
