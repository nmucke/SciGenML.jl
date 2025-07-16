"""
    Preprocessing

Module for preprocessing data.
"""

module Preprocessing

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

import Statistics

"""
    Preprocessor

A preprocessor for data.
"""
abstract type Preprocessor end
export Preprocessor

struct DataPreprocessor
    base::Preprocessor
    target::Preprocessor
    field_conditioning::Preprocessor
    transform::Function
    inverse_transform::Function

    function DataPreprocessor(base_scaler, target_scaler, field_conditioning_scaler)
        return new(
            base_scaler,
            target_scaler,
            field_conditioning_scaler,
            data -> (;
                base = base_scaler.transform(data.base),
                target = target_scaler.transform(data.target),
                field_conditioning = field_conditioning_scaler.transform(data.field_conditioning)
            ),
            data -> (;
                base = base_scaler.inverse_transform(data.base),
                target = target_scaler.inverse_transform(data.target),
                field_conditioning = field_conditioning_scaler.inverse_transform(data.field_conditioning)
            )
        )
    end
end

include("scalers.jl")

export MinMaxScaler, StandardScaler

end
