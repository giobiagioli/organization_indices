from scipy import spatial
from scipy.ndimage import label, center_of_mass
from astropy.stats import RipleysKEstimator
import numpy as np
import sys

#The following routine calculate_Lfunction computes the theoretical and observed Besag's L-functions given a 2D binary field of convective/non-convective points and also provides the cloud-to-cloud nearest-neighbor distances for the calculation of I_org/RI_org.

#	INPUT PARAMETERS
#		nx, ny				number of pixels in the x- and y-directions respectively
#		dxy				grid resolution (assumed to be uniform in both directions)
#		domain_x, domain_y		width and height of the observation window (domain) 
#		cnv_idx				2D binary matrix, =1 in convective points, =0 elsewhere
#		rmax				maximum search radius (box size in the discrete case) for the neighbor counting
#		bins				distance/box size bands in which to evaluate the object counts
#		clustering_algo			flag for the application (True) or not (False) of a four-connectivity clustering algorithm to merge aggregates
#		periodic_BCs			flag for the assignment of periodic (True)/open (False) boundary conditions
# 		periodic_zonal			flag for the assignment of periodic boundary conditions in the x-direction and open boundary conditions in the y-direction (True). If False, this is equivalent to periodic_BCs = False
#		binomial			flag for the assumption of binomial (True)/Poisson (False) model as a reference for spatial randomness
#		border_correction		if binomial is False, this corrects for finiteness of the domain in case the Poisson model is assumed (True)
#		edge_mode			if binomial is True, in case of open domains, this specifies the edge correction method to compensate the undercount bias (options 'none', 'besag', see text for details)

#	OUTPUT PARAMETERS
#		ncnv				number of convective points in the scene (upon applying the clustering algorithm if the clustering_algo flag is True)
#		lambd 				spatial density of convective points
#		NNdist				array containing the cloud-to-cloud nearest-neighbor distances (to be used in the calculation of I_org/RI_org)
#		Besag_obs			Besag's L-function derived from the distribution of the ncnv objects in the scene
#		Besag_theor			Besag's L-function theoretically expected in case the ncnv cloud entities were randomly distributed within the domain   	 	

def calculate_Lfunctions(nx, ny, dxy, domain_x, domain_y, cnv_idx, rmax, bins, clustering_algo, periodic_BCs, periodic_zonal, binomial, border_correction, edge_mode):
	
	##EXCLUSION OF CASES FOR WHICH INPUT ARGUMENTS CONFLICT/ARE NOT ACCOUNTED FOR BY THE ROUTINE
	if (periodic_BCs and periodic_zonal) or (border_correction and binomial) or (border_correction and not periodic_BCs):
		print('--------CONFLICTING INPUT OPTIONS--------')
		sys.exit()
	if (edge_mode=='besag' and not binomial):
		print('--------CASE NOT EXAMINED BY THE PRESENT ROUTINE--------')
		#Built-in functions are available for edge corrections in case of random Poisson processes, see https://docs.astropy.org/en/stable/stats/ripley.html 
		sys.exit()
	
	##DETERMINATION OF CLOUD OBJECT NUMBER AND CENTROIDS
	
	#If four-connectivity clustering algorithms are applied, adjacent convective pixels (i.e., sharing a common side) are merged into a single one. If the domain is cyclic, aggregates on either sides of the domain are close to each other and identified as single ones if they are contiguous. If the domain is cyclic in the zonal but not in the meridional direction, this applies along the x axis only.
	if clustering_algo:
		if periodic_BCs:
			
			#periodic continuation of the domain
			mask = np.block([[cnv_idx, cnv_idx, cnv_idx],[cnv_idx, cnv_idx, cnv_idx],[cnv_idx, cnv_idx, cnv_idx]])
			
			#identification of the clusters and computation of their centers of mass. Only the centroids located within the original (inner) domain are retained. 
			labeled_array, num_features = label(mask)
			centroid = np.asarray(center_of_mass(mask, labeled_array, range(1,num_features+1)))
			centroids_updraft = centroid[np.where((centroid[:,0]>=ny) & (centroid[:,0]<2*ny) & (centroid[:,1]>=nx) & (centroid[:,1]<2*nx))]-[ny,nx]
		else:
			if periodic_zonal:
				
				#periodic continuation of the domain along the zonal direction
				mask = np.block([cnv_idx, cnv_idx, cnv_idx])
				labeled_array, num_features = label(mask)
				centroid = np.asarray(center_of_mass(mask, labeled_array, range(1,num_features+1)))
				centroids_updraft = centroid[np.where((centroid[:,1]>=nx) & (centroid[:,1]<2*nx))]-[0,nx]
			else:
				mask = cnv_idx
				labeled_array, num_features = label(mask)
				centroids_updraft = np.asarray(center_of_mass(mask, labeled_array, range(1,num_features+1)))	
	else:
		centroids_updraft = np.argwhere(cnv_idx)

	#Determination of the number of convective points both with and without the clustering algorithm applied. Their average spatial density is then computed. If the clustering algorithm is applied, the resulting number of convective objects is used in the calculation of density
	ncnv_no_algo = np.sum(cnv_idx)
	ncnv = len(centroids_updraft)
	lambd = ncnv/(domain_x*domain_y)
	
	##DETERMINATION OF NEAREST-NEIGHBOR AND ALL-NEIGHBOR DISTANCES AND COUNTING OF NEIGHBORS IN A RANGE OF DISTANCE/BOX SIZE BANDS FOR ESTIMATION OF OBSERVED L-FUNCTION
	
	#Construct the array of all possible points (including duplicates in case of periodic boundaries)
	if periodic_BCs:
		for xoff in [0,nx,-nx]:
			for yoff in [0,-ny,ny]:
				if xoff==0 and yoff==0:
					j9=centroids_updraft.copy()
				else:
					jo=centroids_updraft.copy()
					jo[:,0]+=yoff
					jo[:,1]+=xoff
					j9=np.vstack((j9,jo))
	else:
		if periodic_zonal:
			for xoff in [0,nx,-nx]:
				if xoff==0:
					j9=centroids_updraft.copy()
				else:
					jo=centroids_updraft.copy()
					jo[:,1]+=xoff
					j9=np.vstack((j9,jo))
		else:
			j9=centroids_updraft.copy()
	
	#Initialization of the array of cloud-to-cloud nearest-neighbor distances
	NNdist = np.array([])
		
	#Initialization of the array whose rows represent the neighbor counting over a range of distances/box sizes (binned) for each element of the pattern
	C = np.zeros((len(centroids_updraft), len(bins)))
	
	#Determination of all-neighbor distances from each point of the pattern in the original domain. In case of periodic boundaries, multiple counting is avoided.  
	for k in range(len(centroids_updraft)):
		hist = np.zeros(len(bins))
		
		#The k-th object is the base point and all its neighbors are considered. In case of cyclic boundaries, the possible neighbors are all the points in the periodically continued domain, except for the duplications of the base point itself.
		a = np.delete(j9, list(range(k, j9.shape[0], len(centroids_updraft))), axis=0)
		tree=spatial.cKDTree(a)
		if periodic_BCs:
			dist,ii=tree.query(centroids_updraft[k,:], 9*(ncnv-1))
			
			#Prohibit multiple counting
			indexes = np.sort(np.unique(a[ii]%[ny,nx], return_index=True, axis = 0)[1])
			dist_new = dist[indexes]
			ii_new = ii[indexes]
			dist, ii = dist_new, ii_new
		else:
			if periodic_zonal:
				#No periodic continuation of the domain along the y-axis, only along x-axis				
				dist,ii=tree.query(centroids_updraft[k,:], 3*(ncnv-1))
				indexes = np.sort(np.unique(a[ii]%[ny,nx], return_index=True, axis = 0)[1])
				dist_new = dist[indexes]
				ii_new = ii[indexes]
				dist, ii = dist_new, ii_new				
			else:
				#In case of open domains, no duplications of the domain are performed
				dist,ii=tree.query(centroids_updraft[k,:], ncnv-1)
		
		#Unit conversion from grid pixels to meters
		dist*=dxy
		
		#Storage of nearest-neighbor distances
		NNdist = np.hstack((NNdist, dist[0]))
		
		#If the discrete version of the Besag's function is to be determined, the distances have to be computed on the discrete grid and their zonal and meridional components are considered 
		if binomial:
			dist_binomial = dxy*np.abs((centroids_updraft[k,:]-tree.data[ii]))
			
			#The size of the box surrounding the k-th object and determined by its j-th neighbor is twice the maximum between the zonal and meridional components of the distance d_{kj}  
			size = 2*np.maximum(dist_binomial[:,0], dist_binomial[:,1])
			
			#Only the box sizes shorter than the maximum allowed size are retained
			size = size[size<=rmax]
			
			#Perform the neighbor counting as a function of distance/box size (cumulative sum). The following procedure is adopted in order to have right-closed intervals, i.e., evaluation of the number of neighbors over boxes of size less or equal than a given value. Note that the bulit-in function numpy.histogram takes right-open bins by definition, with the exception of the last one, hence a different procedure is implemented here
			values,counts = np.unique(np.digitize(size, bins=bins, right=True),  return_counts=True)
			hist[values]=counts
			cum_hist = np.cumsum(hist)
			
			#Definition of edge correction strategies for open domains
			if not periodic_BCs:
				if edge_mode == 'besag':
					weights = np.zeros(len(bins))
					if periodic_zonal:
						for i,ir in enumerate(bins/2.):
							if ir>0:
								#The boxes centered at the k-th object are clipped to the domain edges. If periodic_zonal is True, this occurs only along the meridional direction 
								ymax = np.min((centroids_updraft[k,0]*dxy+ir, domain_y))
								ymin = np.max((centroids_updraft[k,0]*dxy-ir, 0))
								#For each distance ir off the k-th base point, computation of the weighting factor as the fractional area of the box of size 2*ir centered at it and contained within the domain								
								weights[i]=2*ir/(ymax-ymin)
					else:
						for i,ir in enumerate(bins/2.):
							if ir>0:
						 		#The boxes are clipped to the domain edges in both the zonal and meridional directions 
								ymax, xmax = np.min(((centroids_updraft[k,:]*dxy+np.array(ir,ir)), np.array([domain_y, domain_x])), axis = 0)
								ymin, xmin = np.max(((centroids_updraft[k,:]*dxy-np.array(ir,ir)), np.array([0,0])), axis = 0)
								weights[i]=(2*ir)**2/((ymax-ymin)*(xmax-xmin))
					#For each possible size of search boxes centered at the k-th convective object, the weighting factors are assigned to the corresponding counting of neighbors over the boxes
					cum_hist = weights*cum_hist					
		else:
			if periodic_BCs or edge_mode == "none":
				#Only the inter-point distances smaller than the maximum allowed one are retained
				dist = dist[dist<rmax]
				values,counts = np.unique(np.digitize(dist, bins=bins, right=True),  return_counts=True)
				hist[values] = counts
				cum_hist = np.cumsum(hist)
		C[k,:] = cum_hist
	
	##DERIVATION OF THE THEORETICAL AND OBSERVED BESAG'S FUNCTIONS
	#Calculation of the mean number of neighbors off any typical point of the pattern as a function of distance/box size. This is by definition the quantity lambda K(r), lambda being the spatial density of points and K(r) the Ripley's function
	mean_count = np.mean(C, axis = 0)
	
	if binomial:
		#To get the simulated Besag's function, the square root of the Ripley's function has to be taken. Note that mean_count = lambda K(r), hence K(r) = mean_count/lambda, where lambda is estimated as (ncnv-1)/(domain_x*domain_y) in order to have an unbiased estimator. Normalization by rmax (ell_max in the text) is performed. This is formula eqn. (20) in the paper.
		Besag_obs = np.sqrt(mean_count*domain_x*domain_y/(ncnv-1))/rmax
		if periodic_BCs:
			if nx!=ny:
				#This is formula eqn. (22) in the paper, normalized by rmax. A simplification similar to eqn. (19) has been performed, which holds under the assumption of reasonable sample sizes.
				Besag_theor = 1./rmax*np.piecewise(bins, [bins<=min(domain_x, domain_y), bins>min(domain_x, domain_y)], [lambda bins: bins, lambda bins: np.sqrt(bins*domain_y-dxy**2)])		
			else:
				#This is formula eqn. (19) in the paper, normalized by rmax
				Besag_theor = bins/rmax
		else:
			Besag_theor = bins/rmax				
	else:
		if periodic_BCs or edge_mode == "none":
			#Same as line 189, but with the factor 1/pi for the derivation of the Besag's function from the Ripley's function. This is formula eqn. (11) in the paper
			Besag_obs = np.sqrt(1/np.pi*mean_count*domain_x*domain_y/(ncnv-1))/rmax	
			if ny!=nx:
				min_rcrit = min(domain_x, domain_y)/2.
				max_rcrit = max(domain_x, domain_y)/2.
				#This is formula eqn. (21) in the paper, normalized by rmax
				Besag_theor = 1/rmax*np.piecewise(bins, [bins<=min_rcrit, np.logical_and(bins>min_rcrit, bins<=max_rcrit), bins>max_rcrit], [lambda bins: np.sqrt((ncnv-1)/ncnv)*bins, lambda bins: np.sqrt((ncnv-1)/ncnv*1./np.pi*(np.pi*bins**2-2*(bins**2*np.arccos(min_rcrit/bins)-min_rcrit*np.sqrt(bins**2-min_rcrit**2)))), lambda bins: np.sqrt((ncnv-1)/ncnv*1./np.pi*(np.pi*bins**2-2*(bins**2*np.arccos(min_rcrit/bins)-min_rcrit*np.sqrt(bins**2-min_rcrit**2))-2*(bins**2*np.arccos(max_rcrit/bins)-max_rcrit*np.sqrt(bins**2-max_rcrit**2))))])
			else:
				if border_correction:
					#This is formula eqn. (18) in the paper, normalized by rmax
					rcrit = domain_x/2.
					Besag_theor = 1/rmax*np.piecewise(bins, [bins<=rcrit, bins>rcrit], [lambda bins: np.sqrt((ncnv-1)/ncnv)*bins, lambda bins: np.sqrt((ncnv-1)/ncnv*1./np.pi*(np.pi*bins**2-4*(bins**2*np.arccos(rcrit/bins)-rcrit*np.sqrt(bins**2-rcrit**2))))])
				else:
					#Formula eqn. (10) in the paper, normalized by rmax
					Besag_theor = bins/rmax	
	
	return ncnv, lambd, NNdist, Besag_obs, Besag_theor
	
##EXAMPLE USAGE FOR CALCULATION OF INDICES I_ORG/RI_ORG.
#A binned range rIorg of distances must be provided to evaluate the theoretical Weibull NNCDF and, given a scene, construct the observed NNCDF. The latter is not computed through the python built-in function numpy.histogram (see line 148)
 
#       ncnv, lambd, NNdist, Besag_obs, Besag_theor = calculate_Lfunctions(nx, ny, dxy, domain_x, domain_y, cnv_idx, rmax, bins, clustering_algo, periodic_BCs, periodic_zonal, binomial, border_correction, edge_mode)
#       NNCDF_ran_Weibull = 1-np.exp(-lambd*np.pi*rIorg**2)
#       values,counts = np.unique(np.digitize(NNdist, bins=rIorg, right=True), return_counts=True)
#       hist = np.zeros(len(rIorg), dtype = int)
#       hist[values] = counts
#       NNPDF = hist/np.sum(hist)
#       NNCDF_sim = np.cumsum(NNPDF)
#       I_org = np.trapz(NNCDF_sim, x = NNCDF_ran_Weibull)
#       RI_org = np.trapz(NNCDF_sim-NNCDF_ran_Weibull, x = NNCDF_ran_Weibull)


##EXAMPLE USAGE FOR CALCULATION OF INDICES L_ORG/DL_ORG

#       ncnv, lambd, NNdist, Besag_obs, Besag_theor = calculate_Lfunctions(nx, ny, dxy, domain_x, domain_y, cnv_idx, rmax, bins, clustering_algo, periodic_BCs, periodic_zonal, binomial, border_correction, edge_mode)
#       L_org = np.trapz(Besag_obs-Besag_theor, x = bins/rmax)
