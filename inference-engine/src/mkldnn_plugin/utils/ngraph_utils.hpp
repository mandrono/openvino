// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <ngraph/variant.hpp>
<<<<<<< HEAD
#include "transformations/rt_info/primitives_priority_attribute.hpp"
=======
>>>>>>> [CPU] Plug-in migration on ngraph initial commit

namespace MKLDNNPlugin {

inline std::string getRTInfoValue(const std::map<std::string, std::shared_ptr<ngraph::Variant>>& rtInfo, std::string paramName) {
    auto it = rtInfo.find(paramName);
    if (it != rtInfo.end()) {
        auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
        return value->get();
    } else {
        return "";
    }
};

<<<<<<< HEAD
inline std::string getPrimitivesPriorityValue(const std::shared_ptr<ngraph::Node> &node) {
    const auto &rtInfo = node->get_rt_info();
    using PrimitivesPriorityWraper = ngraph::VariantWrapper<ngraph::PrimitivesPriority>;

    if (!rtInfo.count(PrimitivesPriorityWraper::type_info.name)) return "";

    const auto &attr = rtInfo.at(PrimitivesPriorityWraper::type_info.name);
    ngraph::PrimitivesPriority pp = ngraph::as_type_ptr<PrimitivesPriorityWraper>(attr)->get();
    return pp.getPrimitivesPriority();
}

=======
>>>>>>> [CPU] Plug-in migration on ngraph initial commit
template <typename T>
inline const std::shared_ptr<T> getNgraphOpAs(const std::shared_ptr<ngraph::Node>& op) {
    auto typedOp = ngraph::as_type_ptr<T>(op);
    if (!typedOp)
<<<<<<< HEAD
        IE_THROW() << "Can't get ngraph node " << op->get_type_name() << " with name " << op->get_friendly_name();
=======
        THROW_IE_EXCEPTION << "Can't get ngraph node " << op->get_type_name() << " with name " << op->get_friendly_name();
>>>>>>> [CPU] Plug-in migration on ngraph initial commit
    return typedOp;
}

}  // namespace MKLDNNPlugin