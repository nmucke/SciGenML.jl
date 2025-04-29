using SciGenML
using Documenter

DocMeta.setdocmeta!(SciGenML, :DocTestSetup, :(using SciGenML); recursive = true)

makedocs(;
    modules = [SciGenML],
    authors = "ntm <nmucke@gmail.com> and contributors",
    sitename = "SciGenML.jl",
    format = Documenter.HTML(;
        canonical = "https://nmucke.github.io/SciGenML.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = ["Home" => "index.md"]
)

deploydocs(; repo = "github.com/nmucke/SciGenML.jl", devbranch = "main")
