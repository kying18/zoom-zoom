cd(@__DIR__)
using Pkg
Pkg.activate(".")
using PkgTemplates
t = Template(user="ewuirl")
t("mpc")
