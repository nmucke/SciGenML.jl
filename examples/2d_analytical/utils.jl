include("config.jl")

"""
    sample_conditional(base_samples, cond_dist)

Sample from the conditional distribution of the base samples.
"""
function sample_conditional(base_sample, cond_dist, num_samples)
    return rand(cond_dist(base_sample), num_samples)
end;

"""
    get_kde_pdf(samples)

Get the KDE pdf of the samples.
"""
function get_kde_pdf(samples)
    kde = KD.kde(transpose(samples));
    pdf = KD.pdf(kde, x_range, y_range);
    return pdf
end

"""
    get_pdf_diagonal(pdf)

Get the diagonal of the pdf.
"""
function get_pdf_diagonal(pdf)
    return [pdf[i, i] for i in 1:size(pdf, 1)];
end

"""
    plot_pdf(pdf, title = "")

Plot the pdf.
"""
function plot_pdf(pdf, title = "")
    Plots.heatmap(x_range, y_range, pdf, color = :viridis, title = title)
    Plots.plot!(x -> x, color = :red, linewidth = 2)
end
