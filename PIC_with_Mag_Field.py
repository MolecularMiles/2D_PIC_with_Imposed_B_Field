import math
from random import random
import matplotlib
from matplotlib import pyplot
import matplotlib.cm
import numpy
import imageio.v2 as imageio
import os
from math import pi
import scipy.linalg
from scipy.linalg import solve, inv
from numpy import matmul








#_-_-_-_-_-Global simulation variables-_-_-_-_-_-_
#The lattice we'll assign charges to.---------
n_x_points = 80
n_y_points = 80

x_spacing = 10**(-10.0)
y_spacing = x_spacing


box_area = (n_x_points*x_spacing)*(n_y_points*y_spacing)
wall_width = 1




permitivitty = 8.85*(10**(-12.0))
#permitivitty = 1

#atom parameters. --------
# We assume an overall neutral plasma, calculated using the n_atoms metric. Ions act as a stable background due to their high mass, evenly distributed
#across the charge lattice points.
n_atoms = 1000
e_charge = -1.602*(10**-19.0)
#mass = 1
mass = 9.109*(10**(-31.0))
e_number_density = n_atoms/box_area
print("number density: " + str(e_number_density))
e_plasma_freq = (((e_number_density*(e_charge**2.0))/(mass*permitivitty))**0.5)/(2*pi) #Hz
n_points = n_x_points*n_y_points

ion_charge = (n_atoms*(-e_charge))/n_points #background ion charge
e_wave = 1.0
boltz = 1.380649*(10**(-23.0))

probe_x = int(n_x_points/2)
probe_y = int(n_y_points/2)
probe_charge = -e_charge*3



#boundary conditions. 


upper_pot = 0
lower_pot = 0
right_pot = 0
left_pot = 0
left_pot = 0



#simulation time parameters
n_steps = 3000
t = 0
dt = 1/e_plasma_freq
dt = 1/(1000*e_plasma_freq)

t_f = n_steps*dt
inc_wave_freq = 20/(dt*n_steps)
inc_wave_freq = 0

print("Plasma frequency: " + str(e_plasma_freq) + " Hz")

#atom data arrays
vel_deviation = 100
positions = numpy.array([])
velocities = numpy.random.normal(0, vel_deviation, size = (n_atoms, 2))
accelerations = numpy.zeros((n_atoms, 2))
i = 0
while i < n_atoms:

    x = (wall_width*x_spacing) + (x_spacing*(n_x_points-(wall_width)))*random()
    y = (wall_width*y_spacing) + (y_spacing*(n_y_points-(wall_width)))*random()
    if (x < float((n_x_points*x_spacing) - (2*wall_width*x_spacing))) & (x > float((wall_width*x_spacing))):
        if (y < float((n_y_points*y_spacing) - (2*wall_width*y_spacing))) & (y > float((wall_width*y_spacing))):
            temp = numpy.array([x])
            i += 1
            temp = numpy.append(temp, y)
            positions = numpy.append(positions, temp)

print("Assigned all positions...")
positions = positions.reshape(n_atoms, 2)
print("Reshaped the position vector...")


#Assigning magnetic field (pointed along z) values to the lattice points.
def B_z_field(x, y, B_o): #mess with the B_z function to impose any functional form you want, but adjust the timestep accordingly. 
    return(B_o) 
 
b_z_lattice = (numpy.ones((n_y_points, n_x_points)))
for i in range(0, n_x_points):
    for j in range(0, n_y_points):
            b_z_lattice[j, i] = B_z_field(x_spacing*(i-(n_x_points/2)), y_spacing*(j+(n_y_points/2)), 0)


#Creating the array to invert and multiply the charge vector in the solution to poisson's equation via finite differences.
#looking at the structure of the finite differnce formula, we see that the main matrix is just combination of two kinds of blocks:

ones_block = numpy.ones((n_x_points - (2*wall_width)))
ones_block = numpy.diag(ones_block)
print("Built the ones matrix...")
mixed_block = numpy.zeros((n_x_points-(2*wall_width)))
mixed_block = numpy.diag(mixed_block)
for i in range(0, (n_x_points-(2*wall_width))):
    mixed_block[i,i] = -4
    if (i > 0) & (i < (n_x_points-(2*wall_width))-1):
        mixed_block[i, i+1] = 1
        mixed_block[i, i-1] = 1
        mixed_block[i+1, i] = 1
        mixed_block[i-1, i] = 1
print("Built the mixed matrix, now composing the final matrix...")
#now we need to stack arrays vertically and horizontally to build our total matrix. 
final_matrix = numpy.empty([])
temp_row = numpy.array([])
temp_zero_matrix = numpy.array([])
for i in range(0, (n_x_points - (2*wall_width))):
    temp_row = numpy.array([])
    if i == 0:
        temp_row = numpy.hstack((mixed_block, ones_block))
        final_matrix = numpy.hstack((temp_row, numpy.zeros((int(n_x_points-(2*wall_width)), int((n_x_points-(2*wall_width))**2.0) - (2*(n_x_points-(2*wall_width)))))))
    if i == (n_x_points-(2*wall_width)-1):
        temp_row = numpy.hstack((numpy.zeros((int(n_x_points-(2*wall_width)), int(int((n_x_points-(2*wall_width))**2.0) - int(2*(n_x_points-(2*wall_width)))))), ones_block))
        temp_row = numpy.hstack((temp_row, mixed_block))
    if i == 1:
        temp_row = numpy.hstack((ones_block, mixed_block))
        temp_row = numpy.hstack((temp_row, ones_block))
        temp_row = numpy.hstack((temp_row, numpy.zeros((int(n_x_points-(2*wall_width)), int(((n_x_points-(2*wall_width))**2.0) - (3*(n_x_points-(2*wall_width))))))))
    if i == ((n_x_points-(2*wall_width))-2):
        temp_row = numpy.hstack((numpy.zeros((int(n_x_points-(2*wall_width)), int(((n_x_points-(2*wall_width))**2.0) - (3*(n_x_points-(2*wall_width)))))), ones_block))
        temp_row = numpy.hstack((temp_row, mixed_block))
        temp_row = numpy.hstack((temp_row, ones_block))
    


    if (i > 1) & (i < ((n_x_points-(2*wall_width))-2)):
        temp_row = numpy.hstack((numpy.zeros((int(n_x_points-(2*wall_width)), int((i-1)*(n_x_points-(2*wall_width))))), ones_block))
        temp_row = numpy.hstack((temp_row, mixed_block))
        temp_row = numpy.hstack((temp_row, ones_block))
        temp_row = numpy.hstack((temp_row, numpy.zeros(((int(n_x_points-(2*wall_width)), int((n_x_points-(2*wall_width))*(n_x_points-(2*wall_width)-i-2)))))))

    if i != 0:
        final_matrix = numpy.vstack((final_matrix, temp_row))
    if i%10 == 0:
        print("Made it to row " + str(i)+ " out of " +str(n_y_points))
print("Built the matrix. Inverting it now...")
print(final_matrix)
final_matrix = inv(final_matrix)
print("Inverted the matrix. Now starting the simulation...")


#_-_-_-_-_-Simulation Functions-_-_-_-_-_-_
#Grid assign should take in the atom list, assign charges to lattice points, and return the new charge lattice.
def grid_assign(atom_positions):
    charge_lattice = (numpy.ones((n_y_points, n_x_points)))*ion_charge
    for i in range(0, n_atoms):
        x_pedestal = atom_positions[i, 0]
        y_pedestal = atom_positions[i, 1]
        x_float_index, x_int_index = math.modf(x_pedestal/x_spacing) #splitting the lattice assignment into integer and float parts
        y_float_index, y_int_index = math.modf(y_pedestal/y_spacing)
        x_forward_weight = x_float_index #defining forwards as away from the bottom left point, backwards as away from the right/up.
        x_backwards_weight = 1 - x_float_index
        y_forwards_weight = y_float_index
        y_backwards_weight = 1 - y_float_index
        #the bottom left point
        x_1 = int(x_int_index) 
        y_1 = int(y_int_index)
        w_1 = x_backwards_weight*y_backwards_weight #weighting the charge assignment by area.
        #extrapolating to the other points. defined moving clockwise from the bottom left point (point 1)
        #point 2
        x_2 = int(x_int_index)
        y_2 = int(y_int_index+1)
        w_2 = x_backwards_weight*y_forwards_weight
        #point 3
        x_3 = int(x_int_index+1)
        y_3 = int(y_int_index+1)
        w_3 = x_forward_weight*y_forwards_weight
        #point 4
        x_4 = int(x_int_index+1)
        y_4 = int(y_int_index) 
        w_4 = x_forward_weight*y_backwards_weight
        """
        print(x_1)
        print(x_3)
        print(y_1)
        print(y_3)
        print(positions)
        """
        charge_lattice[y_1, x_1] += ((w_1*e_charge)/(x_spacing*y_spacing)) 
        charge_lattice[y_2, x_2] += ((w_2*e_charge)/(x_spacing*y_spacing))
        charge_lattice[y_3, x_3] += ((w_3*e_charge)/(x_spacing*y_spacing))
        charge_lattice[y_4, x_4] += ((w_4*e_charge)/(x_spacing*y_spacing))
    #charge_lattice = numpy.flip(charge_lattice, 1)
    #charge_lattice = numpy.flip(charge_lattice, 0)
    """
    for j in range(0, n_y_points):
        for i in range(0, wall_width+1):
            charge_lattice[j, i] = 0.00000001/permitivitty
    """
    charge_lattice[probe_y, probe_x] = probe_charge/(x_spacing*y_spacing)#chucking a charge in there to see if we get debye screening. 
    return(charge_lattice)

#Potential calculation. Takes in the charge lattice, and calculates the potential based on finite differencing. We assume we are in a grounded box (pot = specified at edges.)
#Since I want to shoot waves into this box, I make it so that we can define the edge potentials differently.
#basically solving matrix equation based on O(x^2, y^2) finite difference method. Look at year 3 numerical methods class notes.
 
def calculate_field(charge_lattice, wave_frequency, t_wave):
    charge_lattice = (-1)*charge_lattice[wall_width:(n_x_points-wall_width),wall_width:(n_x_points-wall_width)]*((x_spacing**2.0)/permitivitty)
    for i in range(wall_width, n_x_points-wall_width-1):
        charge_lattice[wall_width,i] -= upper_pot
        charge_lattice[n_y_points-wall_width-2, i] -= lower_pot 
    for j in range(wall_width, n_y_points-wall_width-1):
        charge_lattice[j,wall_width] -= left_pot
        charge_lattice[j, n_x_points-wall_width-2] -= right_pot
    charge_lattice = charge_lattice.flatten()
    potential_lattice = matmul(final_matrix, charge_lattice)

    potential_lattice = potential_lattice.reshape((n_x_points-2, n_y_points-2))
    potential_lattice = numpy.hstack((potential_lattice, (numpy.ones((n_y_points-2, 1)))*right_pot))
    potential_lattice = numpy.hstack(((numpy.ones((n_y_points-2, 1))*left_pot, potential_lattice)))
    potential_lattice = numpy.vstack((potential_lattice, ((numpy.ones((1, n_x_points)))*lower_pot)))
    potential_lattice = numpy.vstack((((numpy.ones((1, n_x_points)))*upper_pot), potential_lattice))
    
    electric_field_x = numpy.zeros((n_x_points, n_y_points))
    electric_field_y = numpy.zeros((n_x_points, n_y_points))


    for i in range(wall_width, n_x_points-wall_width):
        for j in range(wall_width, n_y_points-wall_width):
            electric_field_x[j, i] = (potential_lattice[j, i-1] - potential_lattice[j, i+1])/(2*(x_spacing)) 
            electric_field_y[j, i] = (potential_lattice[j-1, i] - potential_lattice[j+1, i])/(2*(y_spacing))
    charge_lattice = charge_lattice.reshape(n_x_points-2*wall_width, n_y_points-2*wall_width)
    return(electric_field_x, electric_field_y, potential_lattice, charge_lattice)  



#get acceleration will take in the electric field lattice, and the positions of all atoms.
# We assign field from lattice points to each atom by using the same weights we calculated earlier.  
def get_acceleration(magnetic_field, electric_field_x, electric_field_y, atom_positions, atom_speeds):
    atom_accelerations = numpy.array([])
    for i in range(0, n_atoms):    
        x_pedestal = atom_positions[i, 0]
        y_pedestal = atom_positions[i, 1]
        vx_pedestal = atom_speeds[i, 0]
        vy_pedestal = atom_speeds[i, 1]

        x_float_index, x_int_index = math.modf(x_pedestal/x_spacing) #splitting the lattice assignment into integer and float parts
        y_float_index, y_int_index = math.modf(y_pedestal/y_spacing)
        x_int_index = int(x_int_index)
        y_int_index = int(y_int_index)
        x_forwards_weight = x_float_index #defining forwards as away from the bottom left point, backwards as away from the right/up.
        x_backwards_weight = 1 - x_float_index
        y_forwards_weight = y_float_index
        y_backwards_weight = 1 - y_float_index

        #Now handling the acceleration due to electric fields:
        e_x = ((x_backwards_weight*y_backwards_weight*electric_field_x[y_int_index, x_int_index]) + (x_backwards_weight*y_forwards_weight*electric_field_x[y_int_index+1, x_int_index]) + (x_forwards_weight*y_forwards_weight*electric_field_x[y_int_index+1, x_int_index+1]) + (x_forwards_weight*y_backwards_weight*electric_field_x[y_int_index, x_int_index+1]))
        e_y = ((x_backwards_weight*y_backwards_weight*electric_field_y[y_int_index, x_int_index]) + (x_backwards_weight*y_forwards_weight*electric_field_y[y_int_index+1, x_int_index]) + (x_forwards_weight*y_forwards_weight*electric_field_y[y_int_index+1, x_int_index+1]) + (x_forwards_weight*y_backwards_weight*electric_field_y[y_int_index, x_int_index+1]))
        a_x_from_e_x = (e_charge/mass)*e_x
        a_y_from_e_y = (e_charge/mass)*e_y        

        #handling the acceleration due to the imposed magnetic field. Field points out of plane of simulation, so by basic algebra of the lorentz force:
        b = ((x_backwards_weight*y_backwards_weight*magnetic_field[y_int_index, x_int_index]) + (x_backwards_weight*y_forwards_weight*magnetic_field[y_int_index+1, x_int_index]) + (x_forwards_weight*y_forwards_weight*magnetic_field[y_int_index+1, x_int_index+1]) + (x_forwards_weight*y_backwards_weight*magnetic_field[y_int_index, x_int_index+1]))
        a_x_from_b = vy_pedestal*b*(e_charge/mass)
        a_y_from_b = -vx_pedestal*b*(e_charge/mass)               
        
        #adding up all of the forces:
        a_x_pedestal = a_x_from_e_x + a_x_from_b
        a_y_pedestal = a_y_from_e_y + a_y_from_b
        temp = numpy.array(a_x_pedestal)
        temp = numpy.append(temp, a_y_pedestal)
        atom_accelerations = numpy.append(atom_accelerations, temp)
    atom_accelerations = atom_accelerations.reshape(n_atoms, 2)
    return(atom_accelerations) #same format as atom positions. 




#Velocity Verlet algorithm to make everything move. 

def Verlet_Step(atom_position, atom_veloc, atom_accel, dt, wave_freq, t_wave, mag_field):
    for i in range(0, n_atoms):
        atom_veloc[i,0] += 0.5*(dt)*atom_accel[i, 0]
        atom_veloc[i,1] += 0.5*(dt)*atom_accel[i, 1]


    for i in range(0, n_atoms):
        atom_position[i,0] += dt*atom_veloc[i,0]
        atom_position[i,1] += dt*atom_veloc[i,1]

    new_field_x, new_field_y, potential, charges_dist = calculate_field(grid_assign(atom_position), wave_freq, t_wave)
    atom_accel = get_acceleration(mag_field, new_field_x, new_field_y, atom_position, atom_veloc)
    for i in range(0, n_atoms):
        atom_veloc[i, 0] += 0.5*(atom_accel[i, 0])*dt
        atom_veloc[i, 1] += 0.5*(atom_accel[i, 1])*dt
    
    for i in range(0, n_atoms): #Hard Wall BC
        if atom_position[i, 0] >= (n_x_points-(wall_width+1))*x_spacing:
            atom_veloc[i, 0] = -atom_veloc[i, 0]
        
        if atom_position[i, 0] <= (wall_width+1)*x_spacing:
            atom_veloc[i, 0] = -atom_veloc[i, 0]
        
        if atom_position[i, 1] >= (n_y_points-(wall_width+1))*y_spacing:
            atom_veloc[i, 1] = -atom_veloc[i, 1]
        
        if atom_position[i, 1] <= (wall_width+1)*y_spacing:
            atom_veloc[i, 1] = -atom_veloc[i, 1]
    
    return(atom_position, atom_veloc, atom_accel, new_field_x, new_field_y, potential, charges_dist)




#Some debugging that I needed to do:
"""
print("we have initial position vector of size " + str(numpy.shape(positions)))
tester_set = grid_assign(positions)

pyplot.imshow(tester_set)
print("charge grid has dimensions" + str(numpy.shape(tester_set)))

tester_field_x, tester_field_y = calculate_field(tester_set)
print("The field tensor has shape: x: " + str(numpy.shape(tester_field_x)) + " y: " + str(numpy.shape(tester_field_y)))

tester_accelerations = get_acceleration(tester_field_x, tester_field_y, positions)
print("we get acceleration vector of size " + str(numpy.shape(tester_accelerations)) + " should have the same dimensions as: " + str(numpy.shape(positions)))

new_positions, new_velocities, new_accelerations = Verlet_Step(positions, velocities, tester_accelerations, 0.1)
print("new data:")
print("positions now have shape: " + str(numpy.shape(new_positions)) + " , should have " + str(numpy.shape(positions)))
print("velocities now have shape: " + str(numpy.shape(new_velocities)) + " , should have " + str(numpy.shape(velocities)))
print("accelerations now have shape: " + str(numpy.shape(new_accelerations)) + " , should have " + str(numpy.shape(tester_accelerations)))
"""


#Diagnostic Functions. It's not physics unless you measure it. -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
def get_temperature(velocity_array):
    ms_speed = 0
    for i in range(0, n_atoms):
        vx_pedestal = velocity_array[i, 0]
        vy_pedestal = velocity_array[i, 1]
        ms_speed += ((vx_pedestal**2.0) + (vy_pedestal**2.0))/n_atoms
    temperature = mass*ms_speed/(2*boltz)
    return(temperature)

def get_kinetic(velocity_array):
    kinetic = 0
    for i in range(0, n_atoms):
        vx_pedestal = velocity_array[i, 0]
        vy_pedestal = velocity_array[i, 1]
        kinetic += ((vx_pedestal**2.0) + (vy_pedestal**2.0))*0.5*mass
    return(kinetic)

def get_potential_energy(charge_dist, potential_dist):
    energy_matrix = matmul(charge_dist, potential_dist)
    return(sum(energy_matrix))


def debye_distribution(position_array): #looking for the density distribution around the central point.  
    n_debye_bins = 100
    point_counter = 0
    sep_array = numpy.array([])
    debye_bins = numpy.array([])
    debye_vals = numpy.array([])
    ddebye = n_x_points*x_spacing/(2*n_debye_bins)
    for i in range(0, n_atoms):
        x_pedestal = position_array[i, 0]
        y_pedestal = position_array[i, 1]
        sep_array = numpy.append(sep_array, (((((probe_x*x_spacing)-x_pedestal)**2.0)+(((probe_y*y_spacing)-y_pedestal)**2.0))**0.5))
    sep_array.sort()
    for i in range(0, n_debye_bins):
        debye_bins = numpy.append(debye_bins, ddebye*(i+0.5))
        debye_vals = numpy.append(debye_vals, 0)
        for n in range(0, n_atoms):
            if (sep_array[n] >= (ddebye*i)) & (sep_array[n] < (ddebye*(i+1))):
                point_counter += 1
                debye_vals[i] += 1
    
    for i in range(0, n_debye_bins):
        debye_vals[i] = debye_vals[i]/(e_number_density*2*pi*(((i+0.5)*ddebye)*ddebye))
    return(debye_bins, debye_vals)
                


def get_rdf(position_array): #ONLY USE THIS IF YOU HAVE IMPLEMENTED PERIODIC BOUNDARY CONDITIONS, WHICH I HAVEN'T DONE YET. 
    point_counter = 0
    dr = n_x_points*x_spacing/20
    net_rdf = numpy.array([])
    bins_rdf = numpy.array([])
    temp_rdf = numpy.array([])
    probe_indices = numpy.array([0])
    for i in range(0, len(probe_indices)):
        ped_index = probe_indices[i]
        x_pedestal = position_array[ped_index, 0] #w.l.g choosing atoms 0, 10, 20, 30, 40, 50, 60 to be our RDF probes. 
        y_pedestal = position_array[ped_index, 1]
        for i in range(0, n_atoms):
            if i != ped_index:
                sep = (((x_pedestal - position_array[i, 0])**2.0) + ((y_pedestal - position_array[i, 1])**2.0))**0.5
                temp_rdf = numpy.append(temp_rdf, sep)
    temp_rdf.sort()
    for r in range(0, 20):
        bins_rdf = numpy.append(bins_rdf, dr*(r+0.5))
        net_rdf = numpy.append(net_rdf, 0)
        for data_point in temp_rdf:
            if (data_point >= (r*dr)) & (data_point < (dr*(r+1))):
                net_rdf[r] += 1
                point_counter += 1
    for bin in range(0, len(net_rdf)):
        net_rdf[bin] = net_rdf[bin]/(e_number_density*2*pi*((((bin+0.5)*dr)*dr)))
    return(bins_rdf, net_rdf)

def get_x_dist(position_array):
    n_x_bins = 20
    x_bins = numpy.array([])
    x_vals = numpy.array([])
    dx = n_x_points*x_spacing/n_x_bins
    for i in range(0, n_x_bins):
        x_bins = numpy.append(x_bins, (i+0.5)*dx)
        x_vals = numpy.append(x_vals, 0)
        for n in range(0, n_atoms):
            if (position_array[n, 0] >= (i*dx)) & (position_array[n, 0] < ((i+1)*dx)):
                x_vals[i] += 1/(n_atoms/(n_x_points*x_spacing))
    return(x_bins, x_vals)

def get_y_dist(position_array):
    n_y_bins = 20
    y_bins = numpy.array([])
    y_vals = numpy.array([])
    dy = n_y_points*y_spacing/n_y_bins
    for i in range(0, n_y_bins):
        y_bins = numpy.append(y_bins, (i+0.5)*dy)
        y_vals = numpy.append(y_vals, 0)
        for n in range(0, n_atoms):
            if (position_array[n, 1] >= (i*dy)) & (position_array[n, 1] < ((i+1)*dy)):
                y_vals[i] += 1/(n_atoms/(n_y_points*y_spacing))
    return(y_bins, y_vals)

def get_speed_dist(velocity_array):
    speed_array = numpy.array([])
    for i in range(0, n_atoms):
        speed_array = numpy.append(speed_array, ((((velocity_array[i, 0])**2.0) + ((velocity_array[i, 1])**2.0))**0.5))
    speed_array.sort()

    n_speed_bins = 20
    speed_bins = numpy.array([])
    speed_vals = numpy.array([])
    ds = max(speed_array)/n_speed_bins
    for i in range(0, n_speed_bins):
        speed_bins = numpy.append(speed_bins, (i+0.5)*ds)
        speed_vals = numpy.append(speed_vals, 0)
        for n in range(0, n_atoms):
            if (speed_array[n] >= (i*ds)) & (speed_array[n] < ((i+1)*ds)):
                speed_vals[i] += 1/(n_atoms)
    return(speed_bins, speed_vals)






debye = ((permitivitty*boltz*get_temperature(velocities))/(e_number_density*2*(e_charge**2.0)))**0.5
print("debye length: " + str(debye))
print("cell length: " + str(x_spacing))

#Saving the animation as a gif. Used the tutorial from (https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30)
counter = 0
file_names = numpy.array([])
file_names_ex = numpy.array([])
file_names_ey = numpy.array([])
file_names_pot = numpy.array([])
file_names_charges = numpy.array([])
file_names_rdf = numpy.array([])
file_names_nx = numpy.array([])
file_names_ny = numpy.array([])
file_names_speeds = numpy.array([])
file_names_debye = numpy.array([])

sample_rate = 10
while t <= t_f:  
    positions, velocities, accelerations, temp_field_x, temp_field_y, plot_potential, temp_charges = Verlet_Step(positions, velocities, accelerations, dt, inc_wave_freq, t, b_z_lattice)
    #pyplot.plot(positions[:, 0], positions[:, 1], ".", markersize = 2)
    #pyplot.xlim(0, n_x_points*x_spacing)
    #pyplot.ylim(0, n_y_points*y_spacing)

    #filename = "PIC_Wave_frame_" + str(counter) + ".png"
    #file_names = numpy.append(file_names, filename)

    #pyplot.savefig(filename)
    #pyplot.close()
    if counter%sample_rate == 0:

        print()
        plot_float_index, plot_int_index = math.modf(counter/sample_rate)
        plot_int_index = int(plot_int_index)
        bins, rdf = get_rdf(positions)
        nx_bins, nx_dist = get_x_dist(positions)
        ny_bins, ny_dist = get_y_dist(positions)
        s_bins, s_vals = get_speed_dist(velocities)
        d_bins, d_vals = debye_distribution(positions)

        pyplot.plot(d_bins, d_vals, ".")
        pyplot.xlim(0, n_x_points*x_spacing/2)
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_debye = ("debye_dist_fig" + str(plot_int_index) + ".png")
        file_names_debye = numpy.append(file_names_debye, file_name_debye)
        pyplot.savefig(file_name_debye)
        pyplot.close()

        pyplot.plot(s_bins, s_vals, ".")
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_speed = ("speed_dist_fig_" + str(plot_int_index) + ".png")
        file_names_speeds = numpy.append(file_names_speeds, file_name_speed)
        pyplot.savefig(file_name_speed)
        pyplot.close()

        pyplot.plot(bins, rdf, ".")
        pyplot.xlim(0, n_x_points*x_spacing)
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_rdf = ("rdf_fig_" + str(plot_int_index) + ".png")
        file_names_rdf = numpy.append(file_names_rdf, file_name_rdf)
        pyplot.savefig(file_name_rdf)
        pyplot.close()

        pyplot.plot(nx_bins, nx_dist, ".")
        pyplot.xlim(0, n_x_points*x_spacing)
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_nx = ("nx_dist_fig" + str(plot_int_index) + ".png")
        file_names_nx = numpy.append(file_names_nx, file_name_nx)
        pyplot.savefig(file_name_nx)
        pyplot.close()

        pyplot.plot(ny_bins, ny_dist, ".")
        pyplot.xlim(0, n_y_points*y_spacing)
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_ny = ("ny_dist_fig" + str(plot_int_index) + ".png")
        file_names_ny = numpy.append(file_names_ny, file_name_ny)
        pyplot.savefig(file_name_ny)
        pyplot.close()
        
        pyplot.imshow(temp_charges)
        pyplot.colorbar()
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_charge = ("charge_dist_fig_" + str(plot_int_index) + ".png")
        file_names_charges = numpy.append(file_names_charges, file_name_charge)

        pyplot.savefig(file_name_charge)
        pyplot.close()    

        pyplot.imshow(temp_field_x)
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        pyplot.colorbar()
        file_name_x = ("ex_fig_" + str(plot_int_index) + ".png")
        file_names_ex = numpy.append(file_names_ex, file_name_x)
        pyplot.savefig(file_name_x)
        pyplot.close()

        pyplot.imshow(temp_field_y)
        pyplot.colorbar()
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_y = ("ey_fig_" + str(plot_int_index) + ".png")
        file_names_ey = numpy.append(file_names_ey, file_name_y)
        pyplot.savefig(file_name_y)
        pyplot.close()

        pyplot.imshow(plot_potential)
        pyplot.colorbar()
        pyplot.title("Time: " + str(t) + " s, Temperature: " + str(get_temperature(velocities)) + " K")
        file_name_pot = ("pote_" + str(plot_int_index) + ".png")
        file_names_pot = numpy.append(file_names_pot, file_name_pot)
        pyplot.savefig(file_name_pot)
        #pyplot.show()
        pyplot.close()

        print("made it to " + str(counter) + "th step out of " + str(t_f/dt))
    counter += 1
    t = t+dt


with imageio.get_writer('PIC_Charge_Distribution.gif', mode='I') as writer:
    for filename in file_names_charges:
        image = imageio.imread(filename)
        writer.append_data(image)
"""        
with imageio.get_writer('PIC_Wave.gif', mode='I') as writer:
    for filename in file_names:
        image = imageio.imread(filename)
        writer.append_data(image)
"""
with imageio.get_writer('PIC_RDF.gif', mode='I') as writer:
    for filename in file_names_rdf:
        image = imageio.imread(filename)
        writer.append_data(image)
    
with imageio.get_writer('PIC_speed_dist.gif', mode='I') as writer:
    for filename in file_names_speeds:
        image = imageio.imread(filename)
        writer.append_data(image)

with imageio.get_writer('PIC_debye.gif', mode='I') as writer:
    for filename in file_names_debye:
        image = imageio.imread(filename)
        writer.append_data(image)

with imageio.get_writer('PIC_nx.gif', mode='I') as writer:
    for filename in file_names_nx:
        image = imageio.imread(filename)
        writer.append_data(image)

with imageio.get_writer('PIC_ny.gif', mode='I') as writer:
    for filename in file_names_ny:
        image = imageio.imread(filename)
        writer.append_data(image)


with imageio.get_writer('PIC_Wave_Ex.gif', mode='I') as writer:
    for filename in file_names_ex:
        image = imageio.imread(filename)
        writer.append_data(image)
    
with imageio.get_writer('PIC_Wave_Ey.gif', mode='I') as writer:
    for filename in file_names_ey:
        image = imageio.imread(filename)
        writer.append_data(image)

with imageio.get_writer('PIC_Wave_Pot.gif', mode='I') as writer:
    for filename in file_names_pot:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(file_names_charges):
    os.remove(filename)

for filename in set(file_names_ex):
    os.remove(filename)

for filename in set(file_names_ey):
    os.remove(filename)

for filename in set(file_names_pot):
    os.remove(filename)

for filename in set(file_names_rdf):
    os.remove(filename)

for filename in set(file_names_nx):
    os.remove(filename)

for filename in set(file_names_ny):
    os.remove(filename)

for filename in set(file_names_debye):
    os.remove(filename)

for filename in set(file_names_speeds):
    os.remove(filename)
