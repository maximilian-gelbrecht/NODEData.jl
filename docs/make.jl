using NODEData
using Documenter

DocMeta.setdocmeta!(NODEData, :DocTestSetup, :(using NODEData); recursive=true)

makedocs(;
    modules=[NODEData],
    authors="Maximilian Gelbrecht <maximilian.gelbrecht@posteo.de> and contributors",
    repo="https://github.com/maximilian-gelbrecht/NODEData.jl/blob/{commit}{path}#{line}",
    sitename="NODEData.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://maximilian-gelbrecht.github.io/NODEData.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md"
    ],
)

deploydocs(;
    repo="github.com/maximilian-gelbrecht/NODEData.jl",
    devbranch="main",
)
