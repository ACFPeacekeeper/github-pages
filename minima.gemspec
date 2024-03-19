# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name     = "minima"
  spec.version  = "1.0.0"
  spec.authors  = ["Afonso Fernandes"]
  spec.email    = ["afonso.fernandes@tecnico.ulisboa.pt"]

  spec.summary  = "A custom version of the Minima Jekyll theme"
  spec.homepage = "https://github.com/ACFPeacekeeper/github-pages-blog"
  spec.license  = ""

  spec.metadata["plugin_type"] = "theme"

  spec.files = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r!^(assets|_(includes|layouts|sass)/|(README)((\.(txt|md|markdown)|$)))!i)
  end

  spec.add_runtime_dependency "jekyll", ">= 3.5", "< 5.0"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.9"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.1"

  spec.add_development_dependency "bundler"
end
