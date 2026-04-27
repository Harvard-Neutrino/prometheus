#ifndef LI_BIC_H
#define LI_BIC_H

#include <vector>
#include <LeptonInjector/Particle.h>
#include <photospline/splinetable.h>

// Ben Smithers
// benjamin.smithers@mavs.uta.edu

namespace LeptonInjector {

    ///Configuration parameters needed for all injection modes
	struct BasicInjectionConfiguration{
		BasicInjectionConfiguration();
		///Number of events the generator should/did generate
		uint32_t events;
		///Minimum total event energy to inject
		double energyMinimum;
		///Maximum total event energy to inject
		double energyMaximum;
		///Powerlaw index of the energy spectrum to inject
		double powerlawIndex;
		///Minimum azimuth angle at which to inject events
		double azimuthMinimum;
		///Maximum azimuth angle at which to inject events
		double azimuthMaximum;
		///Minimum zenith angle at which to inject events
		double zenithMinimum;
		///Maximum zenith angle at which to inject events
		double zenithMaximum;
		///Type of first particle to be injected in the final state
		Particle::ParticleType finalType1;
		///Type of second particle to be injected in the final state
		Particle::ParticleType finalType2;

		///Radius around the origin within which to target events
		double injectionRadius;
		///Length of the fixed endcaps add to the distance along which to sample interactions
		double endcapLength;

		///Radius of the origin-centered vertical cylinder within which to inject events
		double cylinderRadius;
		///Height of the origin-centered vertical cylinder within which to inject events
		double cylinderHeight;
		
		std::vector<char> crossSectionBlob;
		std::vector<char> totalCrossSectionBlob;

		void setCrossSection(const photospline::splinetable<>& crossSection, const photospline::splinetable<>& totalCrossSection);
	};

}// end namespace LeptonInjector

#endif 
