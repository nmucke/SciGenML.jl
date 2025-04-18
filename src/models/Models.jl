module Models


export GenerativeModel
export ScoreBasedModel

"""
Abstract type for all generative models.
"""
abstract type GenerativeModel end

"""
Score-based generative model.
"""
struct ScoreBasedModel <: GenerativeModel 
    model_name::String
    model_type::String
    model_path::String

    function ScoreBasedModel()
        return new("ScoreBasedModel", "score_based", "models/score_based")
    end
end


end