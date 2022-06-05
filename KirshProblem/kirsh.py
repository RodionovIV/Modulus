from sympy import Symbol, Eq

import modulus
from modulus.hydra import to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_2d import Rectangle, Circle
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.continuous.inferencer.inferencer import PointwiseInferencer
from modulus.key import Key
from modulus.node import Node
from modulus.PDES.linear_elasticity import LinearElasticityPlaneStress

from modulus.tensorboard_utils.plotter import ValidatorPlotter, InferencerPlotter
from modulus.plot_utils.vtk import var_to_polyvtk
from plotter import CustomInferencerPlotter, CustomValidatorPlotter
from parser import parser

@modulus.main(config_path="conf", config_name="conf")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))
    # specify Panel properties
    E = 20.0 * 10 ** 6  # Pa
    nu = 0.3
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))  # Pa
    mu_real = E / (2 * (1 + nu))  # Pa
    mu_c = mu_real
    lambda_ = lambda_ / mu_c  # Dimensionless
    mu = 1.  # Dimensionless

    # make list of nodes to unroll graph on
    le = LinearElasticityPlaneStress(lambda_=lambda_, mu=mu)
    elasticity_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("F")],
        output_keys=[
            Key("u"),
            Key("v"),
            Key("sigma_xx"),
            Key("sigma_yy"),
            Key("sigma_xy"),
        ],
        cfg=cfg.arch.fully_connected,
    )
    nodes = le.make_nodes() + [
        elasticity_net.make_node(name="elasticity_network", jit=cfg.jit)
    ]

    # add constraints to solver
    # make geometry
    x, y, F = Symbol("x"), Symbol("y"), Key("F")
    panel_origin = (-0.5, -0.5)
    panel_dim = (1, 1)

    window_origin = (-0.2, -0.2)
    window_dim = (0.4, 0.4)

    circle_center = (0,0)
    circle_radius = 0.2

    hr_zone_origin = (-0.3, -0.3)
    hr_zone_dim = (0.6, 0.6)

    hr_zone = Rectangle(hr_zone_origin, (hr_zone_origin[0] + hr_zone_dim[0], hr_zone_origin[1] + hr_zone_dim[1]))


    panel = Rectangle(panel_origin, (panel_origin[0] + panel_dim[0], panel_origin[1] + panel_dim[1]))
    window = Circle(circle_center, circle_radius)
    geo = panel - window

    hr_geo = geo & hr_zone


    # Parameterization
    characteristic_length = panel_dim[0]
    characteristic_disp = 0.001
    sigma_normalization = characteristic_length / (mu_real * characteristic_disp)
    #param_ranges = {sigma_hoop: sigma_hoop_range}
    param_ranges = {F: 7e3 * sigma_normalization}

    # bounds
    bounds_x = (panel_origin[0], panel_origin[0] + panel_dim[0])
    bounds_y = (panel_origin[1], panel_origin[1] + panel_dim[1])
    hr_bounds_x = (hr_zone_origin[0], hr_zone_origin[0] + hr_zone_dim[0])
    hr_bounds_y = (hr_zone_origin[1], hr_zone_origin[1] + hr_zone_dim[1])

    # make domain
    domain = Domain()

    # left wall
    panel_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_left,
        criteria=Eq(x, panel_origin[0]),
        param_ranges=param_ranges,
    )
    domain.add_constraint(panel_left, "panel_left")

    # right wall
    panel_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_right,
        criteria=Eq(x, panel_origin[0] + panel_dim[0]),
        param_ranges=param_ranges,
    )
    domain.add_constraint(panel_right, "panel_right")

    # bottom wall
    panel_bottom = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u":0.0,"v": 0.0},
        lambda_weighting={"u":10.0,"v":10.0},
        batch_size=cfg.batch_size.panel_bottom,
        criteria=Eq(y, panel_origin[1]),
        param_ranges=param_ranges,
    )
    domain.add_constraint(panel_bottom, "panel_bottom")

    # top wall
    panel_top = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": F},
        batch_size=cfg.batch_size.panel_top,
        criteria=Eq(y, panel_origin[1] + panel_dim[1]),
        param_ranges=param_ranges,
    )
    domain.add_constraint(panel_top, "panel_top")

    # pannel window
    panel_window = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=window,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_window,
        param_ranges=param_ranges,
    )
    domain.add_constraint(panel_window, "panel_window")

    # low-resolution interior
    lr_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_xy": 0.0,
        },
        batch_size=cfg.batch_size.lr_interior,
        bounds={x: bounds_x, y: bounds_y},
        lambda_weighting={
            "equilibrium_x": geo.sdf,
            "equilibrium_y": geo.sdf,
            "stress_disp_xx": geo.sdf,
            "stress_disp_yy": geo.sdf,
            "stress_disp_xy": geo.sdf,
        },
        param_ranges=param_ranges,
    )
    domain.add_constraint(lr_interior, "lr_interior")

    # high-resolution interior
    hr_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=hr_geo,
        outvar={
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_xy": 0.0,
        },
        batch_size=cfg.batch_size.hr_interior,
        bounds={x: hr_bounds_x, y: hr_bounds_y},
        lambda_weighting={
            "equilibrium_x": geo.sdf,
            "equilibrium_y": geo.sdf,
            "stress_disp_xx": geo.sdf,
            "stress_disp_yy": geo.sdf,
            "stress_disp_xy": geo.sdf,
        },
        param_ranges=param_ranges,
    )
    domain.add_constraint(hr_interior, "hr_interior")

    # add inferencer data
    invar_numpy = geo.sample_interior(
        10000, bounds={x: bounds_x, y: bounds_y}, param_ranges=param_ranges
    )

    point_cloud_inference = PointwiseInferencer(
        {'x':invar_numpy['x'],'y':invar_numpy['y'],'F':invar_numpy['F']},
        ["u", "v", "sigma_xx", "sigma_yy", "sigma_xy"], nodes, batch_size=4096, plotter=CustomInferencerPlotter(),
    )
    domain.add_inferencer(point_cloud_inference, "inf_data")

    # #Validator
    filename='yyyyy.vtk'
    invar, outvar=parser(filename,sigma_normalization,characteristic_disp)
    invar['F']=invar_numpy['F'][:len(invar['x'])]


    # print(invar)
    validator = PointwiseValidator(
        invar, outvar, nodes, batch_size=128,plotter=CustomValidatorPlotter()
    )
    domain.add_validator(validator)

    #add inferencer data

    grid_inference = PointwiseInferencer(
        invar,
        [
            "u",
            "v",
            "sigma_xx",
            "sigma_yy",
            "sigma_xy",
        ],
        nodes,
        batch_size=128,
    )
    domain.add_inferencer(grid_inference, "grid_inf_data")

    #make solver
    slv = Solver(cfg, domain)

    #from validator import validator
    #validator('yyyyy.vtk','./outputs/kirsh/inferencers/grid_inf_data.vtp',sigma_normalization,characteristic_disp)
    #start solver
    slv.solve()


if __name__ == "__main__":
    run()
