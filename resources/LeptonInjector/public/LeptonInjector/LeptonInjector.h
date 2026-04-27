#ifndef LEPTONINJECTOR_H_INCLUDED
#define LEPTONINJECTOR_H_INCLUDED

#include <queue>
#include <LeptonInjector/DataWriter.h>
#include <earthmodel-service/EarthModelService.h>
#include <phys-services/LICrossSection.h>

#include <photospline/splinetable.h>
#include <photospline/cinter/splinetable.h>
#include <photospline/bspline.h>

#include <iostream>
#include <memory> // adds shared pointer 

#include <LeptonInjector/Coordinates.h>
#include <LeptonInjector/Constants.h>
#include <LeptonInjector/Particle.h>
#include <LeptonInjector/Random.h>
#include <LeptonInjector/EventProps.h>
#include <LeptonInjector/BasicInjectionConfiguration.h>


namespace LeptonInjector{
	
	// Generator configuration structures

	
	///Parameters for injectors placed within a MultiLeptonInjector
	struct Injector{
		// The below  puts in a constructor for the Injector. 
		// you just pass the things in order and BAM
		Injector(unsigned int events,
		  Particle::ParticleType finalType1, Particle::ParticleType finalType2,
		  const std::string& crossSectionPath, const std::string& totalCrossSectionPath,
		  bool ranged):
		events(events),finalType1(finalType1),finalType2(finalType2),
		crossSectionPath(crossSectionPath),totalCrossSectionPath(totalCrossSectionPath),
		ranged(ranged){}
		
		
		///Number of events the generator should/did generate
		unsigned int events;
		///Type of first particle to be injected in the final state
		Particle::ParticleType finalType1;
		///Type of second particle to be injected in the final state
		Particle::ParticleType finalType2;
		///
		std::string crossSectionPath;
		///
		std::string totalCrossSectionPath;
		///
		bool ranged;
	};
	bool operator == (const Injector& one , const Injector& two);
	//----
	
	
	
	//----
	
	class LeptonInjectorBase {
	public:
		LeptonInjectorBase();
		//LeptonInjectorBase(BasicInjectionConfiguration& config, std::shared_ptr<LI_random> random_);
		//No implementation of DAQ; this base class should be pure virtual

		virtual bool Generate()=0;
		//Whether this module has generated as many events already as it was configured to
		virtual std::string Name() const{return("BasicInjector");}
		virtual bool isRanged() const{ return(false); }

		void Print_Configuration();
		void Configure(Injector basic);//, std::shared_ptr<LI_random> pass);

		std::shared_ptr<DataWriter> writer_link;

		BasicInjectionConfiguration& getConfig( void );

	protected:
		///Add common I3Module parameters
		//void AddBaseParameters();
		
		///Get common I3Module parameter values
		
		///Sample a random position on a disk with a given size and orientation.
		///The disk is always centered on the origin of the coordinate system.
		///\param radius the radius of the disk
		///\param zenith the zenith angle of the disk normal
		///\param azimuth the azimuth angle of the disk normal
		LI_Position SampleFromDisk(double radius, double zenith=0., double azimuth=0.);
		
		///Sample one energy value from the energy spectrum
		double SampleEnergy();
		
		///Determine the angles of the final state particles with respect to the
		///initial state neutrino direction, in the lab frame
		///\param E_total the energy of the initial neutrino
		///\param x Bjorken x for the interaction
		///\param y Bjorken y for the interaction
		///\return the relative zenith angles for the first and second final state particles
		std::pair<double,double> computeFinalStateAngles(double E_total, double x, double y);
		
		///\brief Construct an I3MCTree representing an interaction
		///
		///Samples a suitable final state and computes all resulting directions
		///and energies.
		///\param vertex the point at which the interaction occurs
		///\param dir the direction of the interacting neutrino
		///\param energy the energy of the interacting neutrino
		///\param properties the associated structure where the event properties should be recorded
		void FillTree(LI_Position vertex, LI_Direction dir, double energy, BasicEventProperties& properties, std::array<h5Particle,3>& particle_tree);
		
		///Random number source
		///Configuration structure in which to store parameters
		BasicInjectionConfiguration config;
		///Number of events produced so far
		unsigned int eventsGenerated;
		///Whether an S frame has been written
		bool wroteConfigFrame;
		///Whether to suspend the tray after all events have been generated
		bool suspendOnCompletion;
		///The type of interacting neutrino this instance will produce.
		///Note that in the presence of oscillations this may not be the type of
		///the neutrino which arrived at the surface of the Earth.
		Particle::ParticleType initialType;
		
		std::shared_ptr<LI_random> random;

	private:
		LICrossSection crossSection;
		
	};
	
	class RangedLeptonInjector : public LeptonInjectorBase{
	public:
		RangedLeptonInjector();
		RangedLeptonInjector(BasicInjectionConfiguration config, std::shared_ptr<earthmodel::EarthModelService> earth, std::shared_ptr<LI_random> random_);
		bool Generate() override;
		std::string Name() const override {return("RangedInjector");}
		bool isRanged() const override {return(true);}

		// the earthmodel will just be a null poitner at instantiation
		std::shared_ptr<earthmodel::EarthModelService> earthModel;
	
	};
	
	class VolumeLeptonInjector : public LeptonInjectorBase{
	public:
		VolumeLeptonInjector();
		VolumeLeptonInjector(BasicInjectionConfiguration config, std::shared_ptr<earthmodel::EarthModelService> earth, std::shared_ptr<LI_random> random_);
		bool Generate() override;
		std::string Name() const override {return("VolumeInjector");}
		bool isRanged() const override{return(false);}

		// the earthmodel will just be a null poitner at instantiation
		std::shared_ptr<earthmodel::EarthModelService> earthModel;
	};
	
	//----
	
	
	///Construct a new direction with the given relative angles with respect to
	///an existing direction.
	///\param base the existing base direction
	///\param zenith the angle of the new direction with respect to the base
	///\param azimuth the rotation of the new direction about the base
	//std::pair<double,double> rotateRelative(std::pair<double,double> base, double zenith, double azimuth);
	
	
		
	
} //namespace LeptonInjector

#endif
