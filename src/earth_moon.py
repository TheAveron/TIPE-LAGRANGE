from time import sleep

from matplotlib.pyplot import prism
import rebound

def initiator(sim: rebound.Simulation):
    sim.G= 6.6743e-11
    #sim.add(m=1)#1.989e30)       # Star A
    sim.add(m=5.972e24)#, a=1.)#5.972e24, a=1.) # Planet Aa ("Earth")
    sim.add(m=7.348e22, a=2.564e-6, primary = sim.particles[0])#.348e22, a=2.564e-6, primary=sim.particles[1]) # Moon Aa-1 ("Luna"), orbiting planet Aa
    sim.move_to_com()

    sim.status()

def main(sim: rebound.Simulation):
    #op_sun = rebound.OrbitPlot(sim, particles = [1],  unitlabel="[AU]", color=True, periastron=True)
    #op_sun = rebound.OrbitPlot(sim)
    op_earth_moon = rebound.OrbitPlot(sim,particles= [1], primary=0 , show_primary=True, unitlabel="[AU]", color=True, periastron=True) #fig=op_sun.fig, ax = op_sun.ax,)
    op_earth_moon.fig.savefig(f"plots/start.png")
    for i in range(10):
        print(sim.t)
        sim.integrate(sim.t + 0.1)
        #op_sun.update()
        op_earth_moon.update()
        op_earth_moon.fig.savefig(f"plots/out_{i}.png")
        sim.status()

if __name__ == "__main__":
    sim = rebound.Simulation()
    print(sim.t)

    initiator(sim)
    main(sim)
    sim = None
