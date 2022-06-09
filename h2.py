import pygame
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from pygame.locals import *
import pylab
from pyscf import scf
from pyscf import gto, cc
from pyscf.geomopt.geometric_solver import optimize


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
def morse(r):
	z0=.1026277
	s=.72
	req=2.
	x1=1
	return z0*(np.exp(-2.0*s*(r-req)) - 2.0*x1*np.exp(-s*(r-req)))


def add_plot(valuer,valuepot,rval,e,der):
	fig = pylab.figure(figsize=[7, 4], # Inches
                   dpi=100)        # 100 dots per inch, so the resulting buffer is 400x200 pixels
	fig.patch.set_alpha(0.1)           # make the surrounding of the plot 90% transparent to show what it does
	ax = fig.gca()
	ax.plot(valuer,valuepot)
	dx=0.025
	ax.plot(rval,e,"or")
	if showder:
		ax.plot([rval-dx,rval,rval+dx],[e-der*dx,e,e+der*dx],"-,r")
		dxtext = -5*der
		print("dxtext= "+str(dxtext))
		plt.text(rval+dxtext,e,r"$\vec{F}=-\frac{d}{dr}V$", color="red",size="x-large")
	plt.xlabel("$r(a.u.)$")
	plt.ylabel("$E(a.u.)$")

	canvas = agg.FigureCanvasAgg(fig)
	canvas.draw()
	renderer = canvas.get_renderer()
	raw_data = renderer.buffer_rgba()
	size = canvas.get_width_height()
	surf = pygame.image.frombuffer (raw_data, size, "RGBA")
	screen.blit(surf, (100, 5)) # x, y position on screen
	plt.close()
	return canvas, raw_data


def get_opt(roh):
	h1_pos = [100,150]
	h2_pos = [150,100]
	roh = roh 
	dhx = roh/2
	h1_pos[0] = -dhx
	h1_pos[1] = 0
	h2_pos[0] = +dhx
	h2_pos[1] = 0
	mol_h2 = gto.M(atom = 'H '+str(h1_pos[0])+' '+str(h1_pos[1])+' 0; H '+str(h2_pos[0])+' '+str(h2_pos[1])+' 0', basis = 'augccpvdzdk')
	mol_h2.build()
	rhf_h2 = scf.RHF(mol_h2)
	e_h2 = rhf_h2.kernel()
	print(e_h2)
	#mycc = cc.CCSD(rhf_h2o).run()
	#print('CCSD total energy', mycc.e_tot)
	#et = mycc.ccsd_t()

	mol_eq = optimize(rhf_h2, maxsteps=100)
	print(mol_eq.atom_coords())
	roh_eq = mol_eq.atom_coords()[1][0]-mol_eq.atom_coords()[0][0]
	return roh_eq

# https://github.com/uovie/h2opes/blob/master/src/h2opot.f90
def get_pot(roh):
	h1_pos = [100,150]
	h2_pos = [150,100]
	roh = roh 
	dhx = roh/2
	h1_pos[0] = -dhx
	h1_pos[1] = 0
	h2_pos[0] = +dhx
	h2_pos[1] = 0
	mol_h2 = gto.M(atom = 'H '+str(h1_pos[0])+' '+str(h1_pos[1])+' 0; H '+str(h2_pos[0])+' '+str(h2_pos[1])+' 0', basis = 'augccpvdzdk')
	mol_h2.build()
	rhf_h2 = scf.RHF(mol_h2)
	e_h2 = rhf_h2.kernel()
	print(e_h2)
	#mycc = cc.CCSD(rhf_h2o).run()
	#print('CCSD total energy', mycc.e_tot)
	#et = mycc.ccsd_t()
	#print('CCSD(T) total energy', mycc.e_tot + et)

	return e_h2 #mycc.e_tot + et

showder=False
showspring=False
# 25 pixel (size of oxygen) is 0.6 A
pixeltoang = 0.6/25
# 25 pixel (size of oxygen) is 1.13384 bohr
pixeltobohr = 1.13384/25
# 1 A is 1.88973
angtobohr = 1.88973

#h2o equilibrium = 95.7 pm and 104.5 degre
roh=2.0
#roh = get_opt(roh)

pygame.init()
screen = pygame.display.set_mode((800,600))
screen.fill((0, 0, 0))
valuer = np.array([])
valuer = np.append(valuer,roh)
valuepot = np.array([])
valuepot = np.append(valuepot,morse(roh))
canvas, raw_data = add_plot(valuer,valuepot,roh,morse(roh),0)

bg_color = (255, 0, 0)   # fill red as background color
screen.fill(bg_color)
pygame.display.flip()


h1_pos = [100,150]
h2_pos = [150,100]


def draw_h2(center,roh):
	rohorig = roh
	roh = roh / pixeltoang
	mo = 16
	mh = 1
	dhx = roh/2*3
	h1_pos[0] = center[0]-dhx
	h1_pos[1] = center[1] 
	h2_pos[0] = center[0]+dhx
	h2_pos[1] = center[1]
	h1 = pygame.draw.circle(screen,'white',(h1_pos[0],h1_pos[1]),0.53/pixeltoang)
	h2 = pygame.draw.circle(screen,'white',(h2_pos[0],h2_pos[1]),0.53/pixeltoang)
	der=(morse(rohorig+0.001)-morse(rohorig-0.001))/0.002
	print("der")
	print(der)
	longeur = der*5000
	pointe = 15

	if np.abs(der) < 1E-4 and showder:
		print("no arrow")
	elif der > 0 and showder:
		arrow1 = pygame.draw.polygon(screen, (0, 255, 0), ((h1_pos[0], h1_pos[1]), (h1_pos[0]+longeur+pointe, h1_pos[1]), (h1_pos[0]+longeur, h1_pos[1]-pointe), (h1_pos[0]+longeur, h1_pos[1]+pointe),
					     (h1_pos[0]+longeur+pointe, h1_pos[1])))
		arrow2 = pygame.draw.polygon(screen, (0, 255, 0), ((h2_pos[0], h2_pos[1]), (h2_pos[0]-longeur-pointe, h2_pos[1]), (h2_pos[0]-longeur, h2_pos[1]-pointe), (h2_pos[0]-longeur, h2_pos[1]+pointe),
					     (h2_pos[0]-longeur-pointe, h2_pos[1])))
	elif showder:
		arrow1 = pygame.draw.polygon(screen, (0, 255, 0), ((h1_pos[0], h1_pos[1]), (h1_pos[0]+longeur-pointe, h1_pos[1]), (h1_pos[0]+longeur, h1_pos[1]-pointe), (h1_pos[0]+longeur, h1_pos[1]+pointe),
					     (h1_pos[0]+longeur-pointe, h1_pos[1])))
		arrow2 = pygame.draw.polygon(screen, (0, 255, 0), ((h2_pos[0], h2_pos[1]), (h2_pos[0]-longeur+pointe, h2_pos[1]), (h2_pos[0]-longeur, h2_pos[1]-pointe), (h2_pos[0]-longeur, h2_pos[1]+pointe),
					     (h2_pos[0]-longeur+pointe, h2_pos[1])))

	if showspring:
		# test to draw a spring
		debut = [h1_pos[0],h1_pos[1]]
		fin = [h1_pos[0],h1_pos[1]]
		n=2
		x = (h2_pos[0]-h1_pos[0])/60
		y = 7
		while debut[0] < h2_pos[0]:
			fin[0] = debut[0] + x 
			if n < 3:
				fin[1] = debut[1] + y
			elif n < 6 :
				fin[1] = debut[1] - y
			else:
				n = 0
				fin[1] = debut[1] + y
			pygame.draw.line(screen, pygame.Color('orange'), (debut[0], debut[1]), (fin[0], fin[1]), 2)
			debut[0] = fin[0]
			debut[1] = fin[1]
			n = n + 1 
	return der
#screen.blit(circle1)
#def player()
running = True
doh = 0
center = [400,450]
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
			if event.key == pygame.K_SPACE:
				if showder:
					showder = False
				else:
					showder = True
			if event.key == pygame.K_s:
				if showspring:
					showspring = False
				else:
					showspring = True
		if event.type == pygame.KEYUP:
			if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
				doh = 0
	if roh + doh > 0:
		roh += doh
	#print(valuer)
	if roh not in valuer: 
		valuer = np.append(valuer,roh)
		valuepot = np.append(valuepot,morse(roh))
		ind = np.argsort(valuer)
		valuer = valuer[ind]
		valuepot = valuepot[ind]
	der = draw_h2(center,roh)
	add_plot(valuer,valuepot,roh,morse(roh),der)
	#screen.blit(surf, (100, 5)) # x, y position on screen





	pygame.display.update()

	pass
