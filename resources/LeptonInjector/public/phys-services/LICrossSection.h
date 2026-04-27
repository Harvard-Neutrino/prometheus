#ifndef LICrossSection_H
#define LICrossSection_H

#include <photospline/splinetable.h>

#include <LeptonInjector/Particle.h>
#include <LeptonInjector/Random.h>
#include <LeptonInjector/BasicInjectionConfiguration.h>

#include <string>


///Encapsulates cross section information for one physical process with a
///two-body final state.
///
///Cross sections are stored as a pair of photospline objects: one spline for
///the differential cross section, and one for the total cross section. The
///latter could be derived from the former, but is included to avoid users
///having to repeatedly integrate the differential spline numerically.
///
///The differential spline must have three dimensions:
/// - log10 of incoming neutrino energy in GeV
/// - log10 of final state x
/// - log10 of final state y
///The value of the spline must be the log10 of the cross section.
///The absolute value of the differential cross section is never used, so its
///units are not important, however it is advisable that it should be consistent
///with the total cross section (see below).
///
///The total spline must have one dimension, which is log10 of incoming neutrino
///energy in GeV. The value of the spline must be the log10 of the total cross
///section in square meters. 
class LICrossSection{
public:
	LICrossSection(){
		memset(&crossSection,0,sizeof(photospline::splinetable<>));
		memset(&totalCrossSection,0,sizeof(photospline::splinetable<>));
	}
	
	///\param dd_crossSectionFile path to the doubly-differential cross section spline to load
	///\param total_crossSectionFile path to the total cross section file to load
	LICrossSection(std::string dd_crossSectionFile, std::string total_crossSectionFile){
		memset(&crossSection,0,sizeof(photospline::splinetable<>));
		memset(&totalCrossSection,0,sizeof(photospline::splinetable<>));
		load(dd_crossSectionFile, total_crossSectionFile);
	}
	
	///A single final state drawn from a differential cross section
	struct finalStateRecord{
		///Bjorken x
		double x;
		///Bjorken y
		double y;

		///Constructor
		finalStateRecord(double x, double y):
		x(x),y(y){}
	};

	void insert_blobs( LeptonInjector::BasicInjectionConfiguration& config);

	///Sample a final state
	///\param energy the energy of the incoming neutrino
	///\param scatteredType the type of the outgoing lepton
	///\param random a source of random numbers
	finalStateRecord sampleFinalState_DIS(double energy,
	                                  LeptonInjector::Particle::ParticleType scatteredType,
	                                  std::shared_ptr<LeptonInjector::LI_random> random) const;

    
    // the GR sampler just returns X=1
    finalStateRecord sampleFinalState_GR(double energy,
	                                  LeptonInjector::Particle::ParticleType scatteredType,
	                                  std::shared_ptr<LeptonInjector::LI_random> random) const;

    // this has been changed to mainly be a steering function to the other two samplers 
    finalStateRecord sampleFinalState(double energy,
	                                  LeptonInjector::Particle::ParticleType scatteredType,
	                                  std::shared_ptr<LeptonInjector::LI_random> random) const;
	
	///Get the value of the doubly-differential cross section
	///\param energy the neutrino energy in GeV
	///\param x Bjorken x (fraction of the nucleon momentum in the struck parton)
	///\param y Bjorken y (fraction of incoming energy transferred to the nucleon)
	///\param scatteredType the type of the outgoing lepton
	///\return the doubly-differntial cross section in square meters
	double evaluateCrossSection(double energy, double x, double y, 
	                            LeptonInjector::Particle::ParticleType scatteredType) const;
	
	///\param energy the neutrino energy in GeV
	///\return the cross section in in square meters
	double evaluateTotalCrossSection(double energy) const;

	const photospline::splinetable<>& getCrossSection() const{ return(crossSection); }
	const photospline::splinetable<>& getTotalCrossSection() const{ return(totalCrossSection); }

	~LICrossSection(){
		crossSection.~splinetable();
		totalCrossSection.~splinetable();
	}

	///\param dd_crossSectionFile path to the doubly-differential cross section spline to load
	///\param total_crossSectionFile path to the total cross section file to load
	void load(std::string dd_crossSectionFile, std::string total_crossSectionFile);
	
	///\return the minimum value of Q^2 for which the cross section was calculated
	double GetQ2Min() const{ return Q2Min; }
	
	///\return the mass of the target nucleon for which the cross section was calculated
	double GetTargetMass() const{ return targetMass; }
	
	///\return the minimum energy included in this cross section table
	double GetMinimumEnergy() const;
	
	///\return the maximum energy included in this cross section table
	double GetMaximumEnergy() const;
    
    //return the interaction integer 
    int GetInteraction() const{ return interaction; }

private:
	photospline::splinetable<> crossSection;
	photospline::splinetable<> totalCrossSection;
	///The minimum value of Q^2 for which the cross section was calculated
	double Q2Min; 
	///The mass of the target nucleon
	double targetMass;
    ///The interaction type, related to dimensionality of the 
    // 1:CC, 2:NC, 3:GR
    int interaction;
};

#endif //LICrossSection_H
