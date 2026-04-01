#include <I3Test.h>
#include <boost/make_shared.hpp>
#include <boost/python/import.hpp>
#include <phys-services/I3GSLRandomService.h>
#include <icetray/I3Tray.h>
#include <LeptonInjector/LeptonInjector.h>
#include "tools.h"

TEST_GROUP(Utils);

using namespace LeptonInjector;

double particleMass(I3Particle::ParticleType p){
	return(I3Particle(I3Particle::Null,p).GetMass());
}

TEST(compute_kinetic_energy){
	//a particxle of unknown mass should be treated as being massless
	double et=1*I3Units::GeV;
	double ek=kineticEnergy(I3Particle::Hadrons,et);
	ENSURE_EQUAL(ek,et);
	
	//a particle of known mass should have a kinetic energy smaller than the total energy
	ek=kineticEnergy(I3Particle::MuMinus,et);
	ENSURE(ek<et);
	
	//a particle of known mass zero mass should have the same kinetic and
	///total energies
	ek=kineticEnergy(I3Particle::Gamma,et);
	ENSURE_EQUAL(ek,et);
	
	//a total energy smaller than the particle mass should trigger a warning
	//(which we can't verify here) and the kinetic energy should be clipped to zero
	et=1*I3Units::keV;
	ek=kineticEnergy(I3Particle::MuMinus,et);
	ENSURE_EQUAL(ek,0);
}

TEST(compute_particle_speed){
	double ek=1*I3Units::GeV;
	//a particle of unknown mass should be treated as being massless
	//and therefore always have speed c
	double v=particleSpeed(I3Particle::Hadrons, ek);
	ENSURE_EQUAL(v,I3Constants::c);
	
	//particles whose mass is known to be zero should also have speed c
	v=particleSpeed(I3Particle::Gamma, ek);
	ENSURE_EQUAL(v,I3Constants::c);
	
	//a massive particle with kinateic energy equal to its mass should travel at
	//v = sqrt(3/4)*c:
	//Ek = E - m, Ek = m => E = 2m
	//E^2 = p^2 + m^2
	//4 m^2 = p^2 + m^2
	//sqrt(3) m = p
	//p = m gamma beta
	//sqrt(3) = gamma beta
	//3 = beta^2 / (1 - beta^2)
	//3 - 3 beta^2 = beta^2
	//beta = sqrt(3/4)
	//v = c sqrt(3/4)
	v=particleSpeed(I3Particle::MuMinus, particleMass(I3Particle::MuMinus));
	ENSURE_DISTANCE(v,I3Constants::c*sqrt(3.)/2,I3Constants::c/1.e8);
}

TEST(interaction_types){
	//CC
	//nu_e
	ENSURE_EQUAL(deduceInitialType(I3Particle::EMinus,I3Particle::Hadrons),I3Particle::NuE);
	ENSURE_EQUAL(deduceInitialType(I3Particle::EPlus,I3Particle::Hadrons),I3Particle::NuEBar);
	//nu_mu
	ENSURE_EQUAL(deduceInitialType(I3Particle::MuMinus,I3Particle::Hadrons),I3Particle::NuMu);
	ENSURE_EQUAL(deduceInitialType(I3Particle::MuPlus,I3Particle::Hadrons),I3Particle::NuMuBar);
	//nu_tau
	ENSURE_EQUAL(deduceInitialType(I3Particle::TauMinus,I3Particle::Hadrons),I3Particle::NuTau);
	ENSURE_EQUAL(deduceInitialType(I3Particle::TauPlus,I3Particle::Hadrons),I3Particle::NuTauBar);
	//NC
	//nu_e
	ENSURE_EQUAL(deduceInitialType(I3Particle::NuE,I3Particle::Hadrons),I3Particle::NuE);
	ENSURE_EQUAL(deduceInitialType(I3Particle::NuEBar,I3Particle::Hadrons),I3Particle::NuEBar);
	//nu_mu
	ENSURE_EQUAL(deduceInitialType(I3Particle::NuMu,I3Particle::Hadrons),I3Particle::NuMu);
	ENSURE_EQUAL(deduceInitialType(I3Particle::NuMuBar,I3Particle::Hadrons),I3Particle::NuMuBar);
	//nu_tau
	ENSURE_EQUAL(deduceInitialType(I3Particle::NuTau,I3Particle::Hadrons),I3Particle::NuTau);
	ENSURE_EQUAL(deduceInitialType(I3Particle::NuTauBar,I3Particle::Hadrons),I3Particle::NuTauBar);
	//GR
	ENSURE_EQUAL(deduceInitialType(I3Particle::EMinus,I3Particle::NuEBar),I3Particle::NuEBar);
	ENSURE_EQUAL(deduceInitialType(I3Particle::EPlus,I3Particle::NuE),I3Particle::NuEBar);
	ENSURE_EQUAL(deduceInitialType(I3Particle::MuMinus,I3Particle::NuMuBar),I3Particle::NuEBar);
	ENSURE_EQUAL(deduceInitialType(I3Particle::MuPlus,I3Particle::NuMu),I3Particle::NuEBar);
	ENSURE_EQUAL(deduceInitialType(I3Particle::TauMinus,I3Particle::NuTauBar),I3Particle::NuEBar);
	ENSURE_EQUAL(deduceInitialType(I3Particle::TauPlus,I3Particle::NuTau),I3Particle::NuEBar);
	ENSURE_EQUAL(deduceInitialType(I3Particle::Hadrons,I3Particle::Hadrons),I3Particle::NuEBar);
	
	//Other combinations should be rejected
	try{
		deduceInitialType(I3Particle::EPlus,I3Particle::NuMu);
		FAIL("EPlus,NuMu is not a valid interaction final state");
	}catch(...){/*ignore*/}
	try{
		deduceInitialType(I3Particle::Hadrons,I3Particle::TauMinus);
		FAIL("Hadrons,TauMinus is not a valid interaction final state");
	}catch(...){/*ignore*/}
	try{
		deduceInitialType(I3Particle::YAGLaser,I3Particle::PiPlus);
		FAIL("YAGLaser,PiPlus is not a valid interaction final state");
	}catch(...){/*ignore*/}
}

TEST(rotate_relative){
	using LeptonInjector::rotateRelative;
	using I3Constants::pi;
	
	const double tol=1e-8;
	const I3Direction base(2.718,1.414);
	
	//a relative rotation should produce a direction which differs by the input
	//zenith angle, regardless of the input azimuth angle
	for(double zen=0; zen<=3; zen+=.25){
		for(double azi=0; azi<=6; azi+=.25)
			ENSURE_DISTANCE(base.Angle(rotateRelative(base,zen,azi)),zen,tol);
	}
	
	//two relative rotations with azimuth angles which differ by pi should have
	//an angle between them which is twice the rotation zenith angle, unless it
	//is large enough to wrap around, in which case it should be 2pi minus
	//twice the rotation zenith angle
	for(double zen=0; zen<=3; zen+=.25){
		for(double azi=0; azi<=6; azi+=.25){
			I3Direction d1=rotateRelative(base,zen,azi);
			I3Direction d2=rotateRelative(base,zen,(azi<pi ? azi+pi : azi-pi));
			ENSURE_DISTANCE(d1.Angle(d2),(zen<=pi/2 ? 2*zen : 2*(pi-zen)),tol);
		}
	}
}

//Quis probabit ipsa experimenta?
TEST(distribution_moments){
	//a uniform distribution
	double mean=predictPowerLawMoment(0,0,1,1);
	ENSURE_DISTANCE(mean,0.5,1e-6,"Mean of a uniform distribution");
	double variance=predictPowerLawMoment(0,0,1,2);
	ENSURE_DISTANCE(variance,1./12,1e-6,"Variance of a uniform distribution");
	double skewness=predictPowerLawMoment(0,0,1,3)/pow(variance,1.5);
	ENSURE_DISTANCE(skewness,0,1e-6,"Skewness of a uniform distribution");
	double kurtosis=predictPowerLawMoment(0,0,1,4)/(variance*variance) - 3;
	ENSURE_DISTANCE(kurtosis,-6./5,1e-6,"Kurtosis of a uniform distribution");
	
	//an x^0.5 distribution
	mean=predictPowerLawMoment(-.5,3,7.5,1);
	ENSURE_DISTANCE(mean,5.414372,1e-4,"Mean of an x^0.5 distribution");
	variance=predictPowerLawMoment(-.5,3,7.5,2);
	ENSURE_DISTANCE(variance,1.649509,1e-4,"Variance of an x^0.5 distribution");
	skewness=predictPowerLawMoment(-.5,3,7.5,3)/pow(variance,1.5);
	ENSURE_DISTANCE(skewness,-.149699,1e-4,"Skewness of an x^0.5 distribution");
	kurtosis=predictPowerLawMoment(-.5,3,7.5,4)/(variance*variance) - 3;
	ENSURE_DISTANCE(kurtosis,-1.155624,1e-4,"Kurtosis of an x^0.5 distribution");
}