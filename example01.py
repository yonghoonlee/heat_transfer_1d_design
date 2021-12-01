import numpy as np
import openmdao.api as om
import dymos as dm
import time
from matplotlib import pyplot as plt

class heat_transfer_1d_ode(om.ExplicitComponent):

    def __init__(self, *args, **kwargs):
        self.progress_prints = False
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_spatial_nodes', types=int)
        self.options.declare('Stefan_Boltzmann_constant', types=float)

        self.options['num_spatial_nodes'] = 10
        self.options['Stefan_Boltzmann_constant'] = 5.67e-08

    def setup(self):
        nn = self.options['num_nodes']
        ns = self.options['num_spatial_nodes']
        sigma = self.options['Stefan_Boltzmann_constant']

        # Dynamic inputs
        self.add_input('T', val=np.ones((nn, ns), dtype=float), desc='Temperature', units='K')

        # Scalar inputs
        self.add_input('x', val=1.0, desc='Thickness', units='m')
        self.add_input('rho', val=1.0, desc='Material density', units='kg/m**3')
        self.add_input('Cp', val=1.0, desc='Specific heat capacity', units='m**2/K/s**2')
        self.add_input('k', val=1.0, desc='Thermal conductivity', units='kg*m/K/s**3')
        self.add_input('h', val=1.0, desc='Convective heat transfer coefficient', units='kg/K/s**3')
        self.add_input('e', val=1.0, desc='Thermal emittance of the surface', units=None)
        self.add_input('Tinf', val=1.0, desc='Free stream air temperature', units='K')

        # Dynamic outputs
        self.add_output('Tdot', val=np.zeros((nn, ns), dtype=float), desc='Temperature derivative wrt time', units='K/s')

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        ns = self.options['num_spatial_nodes']
        sigma = self.options['Stefan_Boltzmann_constant']

        T = inputs['T']
        x = inputs['x']
        rho = inputs['rho']
        Cp = inputs['Cp']
        k = inputs['k']
        h = inputs['h']
        e = inputs['e']
        Tinf = inputs['Tinf']

        dx = x/(ns - 1)

        Tdot = np.zeros(T.shape, dtype=float)
        Tdot[:,0] = 2.0/(rho*Cp*dx)*(
            h*(Tinf - T[:,0])
#            + e*sigma*1.0e+08*((Tinf/100.0)**4 - (T[:,0]/100.0)**4)
            - k*(T[:,0] - T[:,1])/dx
        )
        Tdot[:,1:(ns-1)] = k/(rho*Cp*dx**2)*(
            T[:,0:(ns-2)] - 2.0*T[:,1:(ns-1)] + T[:,2:ns]
        )
        Tdot[:,(ns-1)] = 2.0*k/(rho*Cp*dx**2)*(
            T[:,ns-2] - T[:,ns-1]
        )

        outputs['Tdot'] = Tdot


def heat_transfer_1d(
        transcription='radau-ps',
        num_segments=72,
        transcription_order=3,
        compressed=True,
        optimizer='IPOPT',
        use_pyoptsparse=True,
        solve_segments='forward'):

    p = om.Problem(model=om.Group())
    if not use_pyoptsparse:
        p.driver = om.ScipyOptimizeDriver()
    else:
        p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer

    if use_pyoptsparse:
        if optimizer == 'SNOPT':
            p.driver.opt_settings['iSumm'] = 6
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['print_level'] = 4

    p.driver.declare_coloring()

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(
            num_segments=num_segments,
            order=transcription_order,
            compressed=compressed,
            solve_segments=solve_segments)
    elif transcription == 'radau-ps':
        t = dm.Radau(
            num_segments=num_segments,
            order=transcription_order,
            compressed=compressed,
            solve_segments=solve_segments)

    phase = dm.Phase(
        ode_class=heat_transfer_1d_ode,
        transcription=t)
    traj.add_phase(name='phase0', phase=phase)

    phase.add_state('T', rate_source='Tdot', targets=['T'], units='K')
    phase.add_parameter('x', targets=['x'], units='m', opt=True, static_target=True)
    phase.add_parameter('rho', targets=['rho'], units='kg/m**3', opt=False, static_target=True)
    phase.add_parameter('Cp', targets=['Cp'], units='m**2/K/s**2', opt=False, static_target=True)
    phase.add_parameter('k', targets=['k'], units='kg*m/K/s**3', opt=False, static_target=True)
    phase.add_parameter('h', targets=['h'], units='kg/K/s**3', opt=False, static_target=True)
    phase.add_parameter('e', targets=['e'], units=None, opt=False, static_target=True)
    phase.add_parameter('Tinf', targets=['Tinf'], units='K', opt=False, static_target=True)

    p.setup()
    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 7200.0)
    p.set_val('traj.phase0.states:T', 373.15)
    p.set_val('traj.phase0.parameters:x', 0.1)
    p.set_val('traj.phase0.parameters:rho', 500.0)
    p.set_val('traj.phase0.parameters:Cp', 2000.0)
    p.set_val('traj.phase0.parameters:k', 0.05)
    p.set_val('traj.phase0.parameters:h', 100.0)
    p.set_val('traj.phase0.parameters:e', 0.9)
    p.set_val('traj.phase0.parameters:Tinf', 2500.0)

    p.run_model()

    t_sol = p.get_val('traj.phase0.timeseries.time')
    T_sol = p.get_val('traj.phase0.timeseries.states:T')
    fg = plt.figure()
    for i in range(0,10):
        plt.plot(t_sol, T_sol[:,i], 'o')
    plt.xlabel('time [s]')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    heat_transfer_1d()




