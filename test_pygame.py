import pygame
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from pygame.locals import *
import pylab
from pyscf import scf
from pyscf import gto, cc


#class pes():
#	def __init__(self,dim):
#		self.dim = dim
#		self.value = 
#	def 
plt.rcParams.update({
    "lines.marker": "o",         # available ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    "lines.linewidth": "1.8",
    "axes.prop_cycle": plt.cycler('color', ['white']),  # line color
    "text.color": "white",       # no text in this example
    "axes.facecolor": "black",   # background of the figure
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",  # no labels in this example
    "axes.grid": "True",
    "grid.linestyle": "--",      # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black", # color surrounding the plot
    "figure.edgecolor": "black",
})




def add_plot(valuer,valuepot):
	fig = pylab.figure(figsize=[4, 2], # Inches
                   dpi=100)        # 100 dots per inch, so the resulting buffer is 400x200 pixels
	fig.patch.set_alpha(0.1)           # make the surrounding of the plot 90% transparent to show what it does
	ax = fig.gca()
	ax.plot(valuer,valuepot)

	canvas = agg.FigureCanvasAgg(fig)
	canvas.draw()
	renderer = canvas.get_renderer()
	raw_data = renderer.buffer_rgba()
	size = canvas.get_width_height()
	surf = pygame.image.frombuffer (raw_data, size, "RGBA")
	screen.blit(surf, (100, 5)) # x, y position on screen
	plt.close()
	return canvas, raw_data


# https://github.com/uovie/h2opes/blob/master/src/h2opot.f90
def get_pot(roh,ra):
	o1_pos = [100,100]
	h1_pos = [100,150]
	h2_pos = [150,100]
	roh = roh 
	dhx = np.sin(ra/2)*roh
	dhy = np.cos(ra/2)*roh
	h1_pos[0] = -dhx
	h1_pos[1] = -dhy
	h2_pos[0] = +dhx
	h2_pos[1] = -dhy
	mol_h2o = gto.M(atom = 'O 0 0 0; H '+str(h1_pos[0])+' '+str(h1_pos[1])+' 0; H '+str(h2_pos[0])+' '+str(h2_pos[1])+' 0', basis = 'augccpvdzdk')
	mol_h2o.build()
	rhf_h2o = scf.RHF(mol_h2o)
	e_h2o = rhf_h2o.kernel()
	print(e_h2o)
	#mycc = cc.CCSD(rhf_h2o).run()
	#print('CCSD total energy', mycc.e_tot)
	#et = mycc.ccsd_t()
	#print('CCSD(T) total energy', mycc.e_tot + et)

	return e_h2o #mycc.e_tot + et

# 25 pixel (size of oxygen) is 0.6 A
pixeltoang = 0.6/25
# 25 pixel (size of oxygen) is 1.13384 bohr
pixeltobohr = 1.13384/25
# 1 A is 1.88973
angtobohr = 1.88973

#h2o equilibrium = 95.7 pm and 104.5 degre

pygame.init()
screen = pygame.display.set_mode((800,600))
screen.fill((0, 0, 0))
valuer = np.array([])
valuer = np.append(valuer,0.957)
valuepot = np.array([])
valuepot = np.append(valuepot,get_pot(0.957,104.5*np.pi/180))
canvas, raw_data = add_plot(valuer,valuepot)

bg_color = (255, 0, 0)   # fill red as background color
screen.fill(bg_color)
pygame.display.flip()


o1_pos = [100,100]
h1_pos = [100,150]
h2_pos = [150,100]


def draw_h2o(center,roh,ra):
	roh = roh / pixeltoang
	mo = 16
	mh = 1
	o1_pos[0] = center[0]
	o1_pos[1] = center[1]
	dhx = np.sin(ra/2)*roh
	dhy = np.cos(ra/2)*roh
	h1_pos[0] = center[0]-dhx
	h1_pos[1] = center[1]-dhy
	h2_pos[0] = center[0]+dhx
	h2_pos[1] = center[1]-dhy
	o1 = pygame.draw.circle(screen,'red',(o1_pos[0],o1_pos[1]),0.6/pixeltoang)
	h1 = pygame.draw.circle(screen,'white',(h1_pos[0],h1_pos[1]),0.53/pixeltoang)
	h2 = pygame.draw.circle(screen,'white',(h2_pos[0],h2_pos[1]),0.53/pixeltoang)
#screen.blit(circle1)
#def player()
running = True
roh=0.957
doh = 0
ra=109*np.pi/180
dra = 0 
center = [400,300]
while running:
	screen.fill((0,0,0))
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			RUNNING = False
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_UP:
				doh += 0.05
				print("doh="+str(doh))
			if event.key == pygame.K_DOWN:
				doh -= 0.05
				print("doh="+str(doh))
			if event.key == pygame.K_LEFT:
				dra += 5*np.pi/180
				valuer = np.array([])
				valuepot = np.array([])
			if event.key == pygame.K_RIGHT:
				dra -= 5*np.pi/180
				valuer = np.array([])
				valuepot = np.array([])
		if event.type == pygame.KEYUP:
			if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
				doh = 0
			if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
				dra = 0
	if roh + doh > 0:
		roh += doh
	ra += dra
	#print(valuer)
	if roh not in valuer: 
		valuer = np.append(valuer,roh)
		valuepot = np.append(valuepot,get_pot(roh,ra))
		ind = np.argsort(valuer)
		valuer = valuer[ind]
		valuepot = valuepot[ind]
	draw_h2o(center,roh,ra)
	add_plot(valuer,valuepot)
	#screen.blit(surf, (100, 5)) # x, y position on screen
	pygame.display.update()

	pass
