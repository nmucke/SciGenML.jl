"""
    Plotting

Module for plotting.

This module contains functions for plotting.
"""

module Plotting

import Plots
import GLMakie

"""
    animate_field(field_list, filename, plot_titles)

Animate a list of fields.
"""
function animate_field(field_list, filename, plot_titles)
    field_list_obs = map(field_list) do field
        GLMakie.Observable(field[:, :, 1])
    end

    # Calculate figure size based on number of plots
    n_plots = length(field_list)
    fig_width = 400 * n_plots # 200 pixels per plot
    fig_height = 400 # Keep height constant
    fig = GLMakie.Figure(; size = (fig_width, fig_height))
    for (i, field_obs) in enumerate(field_list_obs)
        ax = GLMakie.Axis(
            fig[1, i];
            title = plot_titles[i],
            aspect = GLMakie.DataAspect(),
            xticksvisible = false,
            xticklabelsvisible = false,
            yticksvisible = false,
            yticklabelsvisible = false
        )
        GLMakie.heatmap!(ax, field_obs; colormap = GLMakie.Reverse(:Spectral_11))
    end
    ntime = size(field_list[1], 3)
    stream = GLMakie.VideoStream(fig; framerate = 10)
    for i in 1:ntime
        for (field_obs, field) in zip(field_list_obs, field_list)
            field_obs[] = field[:, :, i]
        end
        GLMakie.recordframe!(stream)
    end
    GLMakie.save("$filename.mp4", stream)
end

"""
    animate_velocity_field(field_list, filename, plot_titles; velocity_channels = (1, 2))

Animate a list of velocity fields.
"""
function animate_velocity_magitude(
    field_list,
    filename,
    plot_titles;
    velocity_channels = (1, 2)
)
    field_list = [
        sqrt.(sum(field[:, :, collect(velocity_channels), :] .^ 2, dims = 3)) for
        field in field_list
    ]
    field_list = [
        reshape(field, size(field, 1), size(field, 2), size(field, 4)) for
        field in field_list
    ]
    animate_field(field_list, filename, plot_titles)
end

export animate_field, animate_velocity_magitude

end
