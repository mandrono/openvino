// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_sequence.hpp"

constexpr ngraph::NodeTypeInfo MKLDNNPlugin::RNNSequenceNode::type_info;

MKLDNNPlugin::RNNSequenceNode::RNNSequenceNode(const ngraph::Output<ngraph::Node>& X,
                                               const ngraph::Output<ngraph::Node>& H_t,
                                               const ngraph::Output<ngraph::Node>& sequence_lengths,
                                               const ngraph::Output<ngraph::Node>& W,
                                               const ngraph::Output<ngraph::Node>& R,
                                               const ngraph::Output<ngraph::Node>& B,
                                               std::size_t hidden_size,
                                               ngraph::op::RecurrentSequenceDirection direction,
                                               const std::vector<std::string>& activations,
                                               const std::vector<float>& activations_alpha,
                                               const std::vector<float>& activations_beta,
                                               float clip,
                                               int64_t seq_axis)
    : ngraph::op::util::RNNCellBase({X, H_t, sequence_lengths, W, R, B}, hidden_size, clip, activations, activations_alpha, activations_beta),
          m_direction(direction),
          m_seq_axis(seq_axis) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> MKLDNNPlugin::RNNSequenceNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MKLDNNPlugin::RNNSequenceNode>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
             new_args.at(4), new_args.at(5), m_hidden_size, m_direction, m_activations, m_activations_alpha, m_activations_beta, m_clip, m_seq_axis);
}

void MKLDNNPlugin::RNNSequenceNode::validate_and_infer_types() {
    set_output_size(2);
    ngraph::Shape output_shape_0;
    ngraph::Shape output_shape_1;
    size_t batch_size = get_input_shape(0)[1 - m_seq_axis];
    size_t seq_length = get_input_shape(0)[m_seq_axis];
    size_t num_directions = get_input_shape(1)[1];
    if (m_seq_axis == 1)
        output_shape_0 = ngraph::Shape{batch_size, num_directions, seq_length, m_hidden_size};
    else
        output_shape_0 = ngraph::Shape{seq_length, num_directions, batch_size, m_hidden_size};
    output_shape_1 = ngraph::Shape{batch_size, num_directions, m_hidden_size};
    set_output_type(0, get_input_element_type(0), output_shape_0);
    set_output_type(1, get_input_element_type(0), output_shape_1);
}

bool MKLDNNPlugin::RNNSequenceNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("direction", m_direction);
    visitor.on_attribute("axis", m_seq_axis);
    return ngraph::op::util::RNNCellBase::visit_attributes(visitor);
}
