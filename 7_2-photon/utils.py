
## Utility functions for place cell analysis.
## Code written by Max Jameson Aragon (mjaragon@princeton.edu)
## Last updated 4 March 2021

import numpy as np
import random

def binPosition(x,nBins=25):
    '''Discretize track positions into spatial bins.
    ------
    Inputs
    ------
    x (nx1 array): positions along track
    nBins (int): number of bins to discretize position information
    -------
    Outputs
    -------
    xBins (n-d array): spatial bin for each position sample
    '''
    x = x.flatten() # flatten array
    bins = np.linspace(min(x),max(x),nBins) # position bin values
    xBins = np.digitize(x,bins) # discretize positions
    return xBins

def binDFF(bins,dFF):
    '''Get average dFF value for each spatial bin in track.
    ------
    Inputs
    ------
    bins (n-d array): position bins for each data point
    dFF (n-d array): dFF calcium data
    -------
    Outputs
    -------
    binDFF (m-d array): average dFF value for each spatial bin
    '''
    binIDs = np.unique(bins)
    binDFF = np.array([np.mean(dFF[bins==b]) for b in binIDs])
    return binDFF

def formatToTrials(time,place,dFF):
    '''Format timeseries data into distinct trials for one neuron's data.
    ------
    Inputs
    ------
    time (list-like): time stamps for experiment
    place (list-like): mouse's position in the environment for each time point
    dFF (list-like): neuron's dFF at each time
    -------
    Outputs
    -------
    trialData (nTrials x nBins array): contains dFF data chunked into spatial bins for each trial
    '''

    # We first want to identify the transitions between trials.
    # To do this, we will binarize the position information
    # and find where one trial stops and the next trial starts.

    placeNorm = np.where(place>250,1,0) # binarize data
    oneInds = np.argwhere(placeNorm==1).flatten() + 1 # find where place data are one
    zeroInds = np.argwhere(placeNorm==0).flatten() # find where place data are zero
    trialStarts = np.sort(list(set(oneInds) & set(zeroInds))) # transition points between trials
    trialStops = np.array(trialStarts[1:]) # next transition point
    trialStarts = trialStarts[:-1]

    # Now that we've extracted the trial start and stop indices,
    # let's reformat the data into trials.

    trialData = [] # contains all trials

    # Extract data from each trial
    for start,stop in zip(trialStarts,trialStops):
        trial = dFF[start:stop] # dFF for trial
        trialPositions = place[start:stop] # positions for trial
        bins = binPosition(trialPositions) # chunk the positions into spatial bins
        trialdFF = binDFF(bins,trial) # chunk the dFF data into spatial bins
        trialData.append(trialdFF) # append trial

    trialData = np.array(trialData)

    return trialData

def getPlaceProba(bins):
    '''Get position occupancy probability for each spatial bin.
    ------
    Inputs
    ------
    bins (n-d array): spatial bin ID for each data point
    -------
    Outputs
    -------
    binProbs (m-d array): empirical probability distribution over positions
    '''
    binIDs = np.unique(bins) # unique bins along track
    binCounts = np.array([np.sum(bins==b) for b in binIDs]) # number of instances within each bin
    binProbs = binCounts/sum(binCounts) # place occupancy probability
    return binProbs

def computeSkaggsInfo(place,dFF):
    '''Compute place information given place data and neural activity.
    ------
    Inputs
    ------
    place (list-like): mouse's position in the environment for each time point
    dFF (list-like): trial-formatted dF/F0 data
    -------
    Outputs
    -------
    info (float): Skaggs information for neuron
    '''

    # This function computes the Skaggs information
    getInfo = lambda probs,dFF,meanDFF: sum(probs * dFF * np.log2(dFF/meanDFF))

    # Get average dFF across all trials
    dFF = np.mean(dFF,axis=0)

    # Get spatial bins
    bins = binPosition(place)

    # Get animal's positional occupancy probability
    placeProbs = getPlaceProba(bins)

    # Only consider non-zero dFFs for mutual info calculation
    nonzeroInds = np.argwhere(dFF!=0) # indices where dFF is nonzero
    placeProbs = placeProbs[nonzeroInds] # filtered place probabilites
    dFF = dFF[nonzeroInds] # filtered dFF values

    # Calculate mutual information
    meanDFF = np.sum(placeProbs*dFF) # mean dFF along virtual track
    info = getInfo(placeProbs,dFF,meanDFF) # mutual information

    return info

def getNullDistribution(place,dFF,nRepeats=100):
    '''Generate null distribution of place information by shuffling trials.
    ------
    Inputs
    ------
    place (nx1 array): positions along track
    dFF (n-d array): trial-formatted dF/F0 data
    nRepeats (int): number of samples in null distribution
    -------
    Outputs
    -------
    nullDistribution (nRepeats-d array): null distribution of place information
    '''

    nullDistribution = [] # contains place information from shuffled trials

    # Generate null distribution by shuffling dF/F data for each trial
    for ii in range(nRepeats):
        shuffledData = dFF.copy()
        [random.shuffle(r) for r in shuffledData] # shuffle dFF data within each trial
        info = computeSkaggsInfo(place,shuffledData) # compute Skaggs info for shuffled data
        nullDistribution.extend(info.copy()) # extend null distribution

    nullDistribution = np.array(nullDistribution)

    return nullDistribution
