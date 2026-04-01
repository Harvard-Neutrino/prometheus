#include <I3Test.h>

#include <fstream>

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>

#include <icetray/I3Tray.h>
#include <dataclasses/physics/I3MCTreeUtils.h>
#include <LeptonInjector/LeptonInjector.h>

#include "tools.h"

TEST_GROUP(RangedInjection);

using namespace LeptonInjector;

//None of the default values in RangedInjectionConfiguration should set off
//errors in the module
TEST(1_sane_defaults_in_ranged_config){
	RangedInjectionConfiguration config;
	RangedLeptonInjector inj(context,config);
	ConfigureStandardParams(inj);
}

//attempting to set nonsensical parameter values should be rejected
TEST(2_reject_invalid_ranged_params){
	try{
		RangedInjectionConfiguration config;
		config.events=0;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with 0 events should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.energyMinimum=0;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a non-positive minimum energy should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.energyMaximum=0;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a non-positive maximum energy should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.energyMinimum=20;
		config.energyMaximum=10;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum energy below the minimum energy should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.azimuthMinimum=-2;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative minimum azimuth should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.azimuthMaximum=7;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum azimuth greater than 2 pi should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.azimuthMinimum=2;
		config.azimuthMaximum=1;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum azimuth less than the minimum azimuth should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.zenithMinimum=-2;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative minimum zenith should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.zenithMaximum=4;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum zenith greater than pi should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.zenithMinimum=2;
		config.zenithMaximum=1;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum zenith less than the minimum zenith should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.finalType1=I3Particle::Neutron;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a non-supported particle type should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.finalType2=I3Particle::YAGLaser;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a non-supported particle type should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		I3Context context;
		//no random service
		context.Put(earthmodelService,earthModelName);
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("not finding the random service in the cntext should be detected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		I3Context context;
		context.Put(randomService);
		//no earth model service
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("not finding the Earth model service in the cntext should be detected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		RangedLeptonInjector inj(context,config);
		//don't set cross section file
		inj.GetConfiguration().Set("EarthModel",boost::python::object(earthModelName));
		inj.Configure();
		FAIL("configuring without a cross section should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		RangedLeptonInjector inj(context,config);
		inj.GetConfiguration().Set("DoublyDifferentialCrossSectionFile",boost::python::object(defaultCrosssectionPath));
		inj.GetConfiguration().Set("TotalCrossSectionFile",boost::python::object(defaultTotalCrosssectionPath));
		//don't set earth model service
		inj.Configure();
		FAIL("configuring without an earth model service should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.injectionRadius=-200;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative injection radius should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration config;
		config.endcapLength=-200;
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative endcap length should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
}

TEST(3_number_of_events){
	const unsigned int nEvents=100;
	const std::string filename=I3Test::testfile("Ranged_NEvents_test.i3");
	
	I3Tray tray;
	tray.GetContext().Put(randomService);
	tray.GetContext().Put(earthmodelService,earthModelName);
	tray.AddModule("I3InfiniteSource")("Stream",I3Frame::Stream('Q'));
	tray.AddModule("RangedLeptonInjector")
	("NEvents",nEvents)
	("MinimumEnergy",1e2)
	("MaximumEnergy",1e6)
	("DoublyDifferentialCrossSectionFile",defaultCrosssectionPath)
	("TotalCrossSectionFile",defaultTotalCrosssectionPath)
	("EarthModel",earthModelName);
	tray.AddModule("I3Writer")("Filename",filename);
	tray.Execute();
	
	std::ifstream resultsFile(filename.c_str());
	ENSURE(resultsFile.good()); //the file should exist and be readable
	
	I3Frame f;
	//There should be a TrayInfo frame
	f.load(resultsFile);
	ENSURE(!resultsFile.fail());
	ENSURE(f.GetStop()==I3Frame::Stream('I'));
	//There should be an S frame
	f.load(resultsFile);
	ENSURE(!resultsFile.fail());
	ENSURE(f.GetStop()==I3Frame::Stream('S'));
	//and it should contain the generator properties
	ENSURE(f.Has("LeptonInjectorProperties"));
	
	//Then there should be nEvents Q frames
	for(unsigned int i=0; i<nEvents; i++){
		f.load(resultsFile);
		ENSURE(!resultsFile.fail());
		ENSURE(f.GetStop()==I3Frame::Stream('Q'));
		//each Q frame should have its MCTree and generation properties
		ENSURE(f.Has("I3MCTree"));
		ENSURE(f.Has("EventProperties"));
	}
	
	//There should be no more frames
	f.load(resultsFile);
	ENSURE(resultsFile.fail());
	
	//If the test was successful, clean up
	unlink(filename.c_str());
}

TEST(4_particle_type_production){
	RangedInjectionConfiguration config;
	config.events=2000;
	{ //nu_mu CC
		config.finalType1=I3Particle::MuMinus;
		config.finalType2=I3Particle::Hadrons;
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			ENSURE(primary.GetType()==I3Particle::NuMu,"The primary should have type 'NuMu'");
			std::vector<I3Particle> products=I3MCTreeUtils::GetDaughters(tree,primary);
			ENSURE(products.size()==2,"Two particles should be produced at the interaction vertex");
			ENSURE((products[0].GetType()==config.finalType1 && products[1].GetType()==config.finalType2)
				   || (products[1].GetType()==config.finalType1 && products[0].GetType()==config.finalType2));
		}
	}
	
	{ //nu_tau NC
		config.finalType1=I3Particle::NuTau;
		config.finalType2=I3Particle::Hadrons;
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			ENSURE(primary.GetType()==I3Particle::NuTau,"The primary should have type 'NuTau'");
			std::vector<I3Particle> products=I3MCTreeUtils::GetDaughters(tree,primary);
			ENSURE(products.size()==2,"Two particles should be produced at the interaction vertex");
			ENSURE((products[0].GetType()==config.finalType1 && products[1].GetType()==config.finalType2)
				   || (products[1].GetType()==config.finalType1 && products[0].GetType()==config.finalType2));
		}
	}
	
	{ //nu_e^bar GR
		config.finalType1=I3Particle::EPlus;
		config.finalType2=I3Particle::NuE;
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			ENSURE(primary.GetType()==I3Particle::NuEBar,"The primary should have type 'NuEBar'");
			std::vector<I3Particle> products=I3MCTreeUtils::GetDaughters(tree,primary);
			ENSURE(products.size()==2,"Two particles should be produced at the interaction vertex");
			ENSURE((products[0].GetType()==config.finalType1 && products[1].GetType()==config.finalType2)
				   || (products[1].GetType()==config.finalType1 && products[0].GetType()==config.finalType2));
		}
	}
}

TEST(5_energy_distribution){
	resetRandomState();
	RangedInjectionConfiguration config;
	config.events=50000;
	{ //uniform distribution
		config.energyMinimum=100*I3Units::GeV;
		config.energyMaximum=1000*I3Units::GeV;
		config.powerlawIndex=0;
		
		MomentAccumulator moments;
		double minEnSeen=1e10*I3Units::GeV, maxEnSeen=0*I3Units::GeV;
		
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			minEnSeen=std::min(minEnSeen,props.totalEnergy);
			maxEnSeen=std::max(maxEnSeen,props.totalEnergy);
			moments.Insert(props.totalEnergy);
		}
		
		//check the range
		ENSURE(minEnSeen>=config.energyMinimum,"Sampled energies should be in bounds");
		ENSURE(maxEnSeen<=config.energyMaximum,"Sampled energies should be in bounds");
		ENSURE_DISTANCE(minEnSeen,config.energyMinimum,1*I3Units::GeV,
						"Minimum energy should be approached"); //P(Fail)=3.9e-3
		ENSURE_DISTANCE(maxEnSeen,config.energyMaximum,1*I3Units::GeV,
						"Maximum energy should be approached"); //P(Fail)=3.9e-3
		//check the shape
		testPowerLawness(config.powerlawIndex, config.energyMinimum, config.energyMaximum, config.events,
						 moments, __FILE__, __LINE__);
	}
	
	{ //E^-1
		config.energyMinimum=100*I3Units::GeV;
		config.energyMaximum=1000*I3Units::GeV;
		config.powerlawIndex=1;
		
		MomentAccumulator moments;
		double minEnSeen=1e10*I3Units::GeV, maxEnSeen=0*I3Units::GeV;
		
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			minEnSeen=std::min(minEnSeen,props.totalEnergy);
			maxEnSeen=std::max(maxEnSeen,props.totalEnergy);
			moments.Insert(props.totalEnergy);
		}
		
		//check the range
		ENSURE(minEnSeen>=config.energyMinimum,"Sampled energies should be in bounds");
		ENSURE(maxEnSeen<=config.energyMaximum,"Sampled energies should be in bounds");
		ENSURE_DISTANCE(minEnSeen,config.energyMinimum,5*I3Units::GeV,
						"Minimum energy should be approached"); //P(Fail)=~4e-10
		ENSURE_DISTANCE(maxEnSeen,config.energyMaximum,5*I3Units::GeV,
						"Maximum energy should be approached"); //P(Fail)=1.9e-5
		//check the shape
		testPowerLawness(config.powerlawIndex, config.energyMinimum, config.energyMaximum, config.events,
						 moments, __FILE__, __LINE__);
	}
	
	{ //E^-2
		config.energyMinimum=100*I3Units::GeV;
		config.energyMaximum=1000*I3Units::GeV;
		config.powerlawIndex=2;
		
		MomentAccumulator moments;
		double minEnSeen=1e10*I3Units::GeV, maxEnSeen=0*I3Units::GeV;
		
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			minEnSeen=std::min(minEnSeen,props.totalEnergy);
			maxEnSeen=std::max(maxEnSeen,props.totalEnergy);
			moments.Insert(props.totalEnergy);
		}
		
		//check the range
		ENSURE(minEnSeen>=config.energyMinimum,"Sampled energies should be in bounds");
		ENSURE(maxEnSeen<=config.energyMaximum,"Sampled energies should be in bounds");
		ENSURE_DISTANCE(minEnSeen,config.energyMinimum,5*I3Units::GeV,
						"Minimum energy should be approached"); //P(Fail)=~4e-10
		ENSURE_DISTANCE(maxEnSeen,config.energyMaximum,5*I3Units::GeV,
						"Maximum energy should be approached"); //P(Fail)=1.9e-5
		//check the shape
		testPowerLawness(config.powerlawIndex, config.energyMinimum, config.energyMaximum, config.events,
						 moments, __FILE__, __LINE__);
	}
	
	{ //E^1
		config.energyMinimum=100*I3Units::GeV;
		config.energyMaximum=1000*I3Units::GeV;
		config.powerlawIndex=-1;
		
		MomentAccumulator moments;
		double minEnSeen=1e10*I3Units::GeV, maxEnSeen=0*I3Units::GeV;
		
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			minEnSeen=std::min(minEnSeen,props.totalEnergy);
			maxEnSeen=std::max(maxEnSeen,props.totalEnergy);
			moments.Insert(props.totalEnergy);
		}
		
		//check the range
		ENSURE(minEnSeen>=config.energyMinimum,"Sampled energies should be in bounds");
		ENSURE(maxEnSeen<=config.energyMaximum,"Sampled energies should be in bounds");
		ENSURE_DISTANCE(minEnSeen,config.energyMinimum,5*I3Units::GeV,
						"Minimum energy should be approached"); //P(Fail)=~4e-10
		ENSURE_DISTANCE(maxEnSeen,config.energyMaximum,5*I3Units::GeV,
						"Maximum energy should be approached"); //P(Fail)=1.9e-5
		//check the shape
		testPowerLawness(config.powerlawIndex, config.energyMinimum, config.energyMaximum, config.events,
						 moments, __FILE__, __LINE__);
	}
}

TEST(6_zenith_distribution){
	RangedInjectionConfiguration config;
	config.events=10000;
	{ //standard zenith range
		MomentAccumulator moments;
		double minCosZenSeen=2, maxCosZenSeen=-2;
		
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			double cosZen=cos(props.zenith);
			minCosZenSeen=std::min(minCosZenSeen,cosZen);
			maxCosZenSeen=std::max(maxCosZenSeen,cosZen);
			moments.Insert(cosZen);
		}
		
		//check the range
		ENSURE(acos(maxCosZenSeen)>=config.zenithMinimum,"Sampled cosine zeniths should be in bounds");
		ENSURE(acos(minCosZenSeen)<=config.zenithMaximum,"Sampled cosine zeniths should be in bounds");
		ENSURE_DISTANCE(maxCosZenSeen,cos(config.zenithMinimum),.005,
						"Maximum cos(zenith) should be approached");
		ENSURE_DISTANCE(minCosZenSeen,cos(config.zenithMaximum),.005,
						"Minimum cos(zenith) should be approached");
		//check the shape
		testPowerLawness(0, cos(config.zenithMaximum), cos(config.zenithMinimum), config.events,
						 moments, __FILE__, __LINE__);
	}
	
	{ //some other zenith range
		MomentAccumulator moments;
		config.zenithMinimum=.3;
		config.zenithMaximum=1.8;
		double minCosZenSeen=2, maxCosZenSeen=-2;
		
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			double cosZen=cos(props.zenith);
			minCosZenSeen=std::min(minCosZenSeen,cosZen);
			maxCosZenSeen=std::max(maxCosZenSeen,cosZen);
			moments.Insert(cosZen);
		}
		
		//check the range
		ENSURE(acos(maxCosZenSeen)>=config.zenithMinimum,"Sampled cosine zeniths should be in bounds");
		ENSURE(acos(minCosZenSeen)<=config.zenithMaximum,"Sampled cosine zeniths should be in bounds");
		ENSURE_DISTANCE(maxCosZenSeen,cos(config.zenithMinimum),.005,
						"Maximum cos(zenith) should be approached");
		ENSURE_DISTANCE(minCosZenSeen,cos(config.zenithMaximum),.005,
						"Minimum cos(zenith) should be approached");
		//check the shape
		testPowerLawness(0, cos(config.zenithMaximum), cos(config.zenithMinimum), config.events,
						 moments, __FILE__, __LINE__);
	}
}

TEST(7_azimuth_distribution){
	RangedInjectionConfiguration config;
	config.events=10000;
	{ //standard azimuth range
		MomentAccumulator moments;
		double minAziSeen=7, maxAziSeen=-1;
		
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			minAziSeen=std::min(minAziSeen,props.azimuth);
			maxAziSeen=std::max(maxAziSeen,props.azimuth);
			moments.Insert(props.azimuth);
		}
		
		//check the range
		ENSURE(minAziSeen>=config.azimuthMinimum,"Sampled azimuths should be in bounds");
		ENSURE(maxAziSeen<=config.azimuthMaximum,"Sampled azimuths should be in bounds");
		ENSURE_DISTANCE(minAziSeen,config.azimuthMinimum,.01,
						"Minimum azimuth should be approached");
		ENSURE_DISTANCE(maxAziSeen,config.azimuthMaximum,.01,
						"Maximum azimuth should be approached");
		//check the shape
		testPowerLawness(0, config.azimuthMinimum, config.azimuthMaximum, config.events,
						 moments, __FILE__, __LINE__);
	}
	
	{ //some other azimuth range
		MomentAccumulator moments;
		config.azimuthMinimum=1;
		config.azimuthMaximum=5;
		double minAziSeen=7, maxAziSeen=-1;
		
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			minAziSeen=std::min(minAziSeen,props.azimuth);
			maxAziSeen=std::max(maxAziSeen,props.azimuth);
			moments.Insert(props.azimuth);
		}
		
		//check the range
		ENSURE(minAziSeen>=config.azimuthMinimum,"Sampled azimuths should be in bounds");
		ENSURE(maxAziSeen<=config.azimuthMaximum,"Sampled azimuths should be in bounds");
		ENSURE_DISTANCE(minAziSeen,config.azimuthMinimum,.01,
						"Minimum azimuth should be approached");
		ENSURE_DISTANCE(maxAziSeen,config.azimuthMaximum,.01,
						"Maximum azimuth should be approached");
		//check the shape
		testPowerLawness(0, config.azimuthMinimum, config.azimuthMaximum, config.events,
						 moments, __FILE__, __LINE__);
	}
}

TEST(8_final_state_distribution){
	//incoming lepton energy for this test
	const double energy=1e3;
	I3CrossSection xs(defaultCrosssectionPath,defaultTotalCrosssectionPath);
	//figure out the valid domain for output
	const double s=pow((xs.GetTargetMass()+energy),2)-pow(energy,2);
	const double logxyMin=log10(xs.GetQ2Min()/s);
	//set up a quick-and-dirty histogram
	const unsigned int divisions=20; //binning in log(x) and log(y);
	const double binWidth=(0-logxyMin)/divisions;
	unsigned int counts[divisions][divisions];
	std::fill_n(&counts[0][0],divisions*divisions,0);
	
	//generate samples and histogram them
	RangedInjectionConfiguration config;
	config.events=1000000;
	config.energyMinimum=energy*I3Units::GeV;
	config.energyMaximum=energy*I3Units::GeV;
	resetRandomState();
	RangedLeptonInjector inj(context,config);
	ConfigureStandardParams(inj);
	boost::shared_ptr<OutputCollector> col=connectCollector(inj);
	while(!inj.DoneGenerating()){
		boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
		inj.DAQ(f);
		const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
		double lx=log10(props.finalStateX);
		double ly=log10(props.finalStateY);
		
		ENSURE(lx>=logxyMin && lx<=0);
		ENSURE(ly>=logxyMin && ly<=0);
		unsigned int xIdx=(lx-logxyMin)/binWidth;
		unsigned int yIdx=(ly-logxyMin)/binWidth;
		counts[xIdx][yIdx]++;
	}
	
	//figure out what we expect by directly integrating the cross section
	double expectations[divisions][divisions];
	{
		typedef double(*gslFuncType)(double*,size_t,void*);
		//VEGAS requires random numbers
		gsl_rng_env_setup();
		const gsl_rng_type* T=gsl_rng_default;
		std::unique_ptr<gsl_rng,void(*)(gsl_rng*)> r(gsl_rng_alloc(T),gsl_rng_free);
		gsl_rng_set(r.get(), 1337);
		double lower[2], upper[2]; //limits of integration
		//context data for the integrand
		struct paramsContainer{
			const I3CrossSection& xs;
			const double energy;
		} params{xs,energy};
		//the integrand
		auto gslWrapper=[](double* k, size_t dim, void* paramsp)->double{
			paramsContainer& params=*static_cast<paramsContainer*>(paramsp);
			double x=pow(10.,k[0]);
			double y=pow(10.,k[1]);
			double jacobian=x*y*log(10.)*log(10.);
			return(jacobian*params.xs.evaluateCrossSection(params.energy,x,y,I3Particle::MuMinus));
		};
		double total=0;
		//compute the intergal over each bin
		for(unsigned int xIdx=0; xIdx<divisions; xIdx++){
			lower[0]=logxyMin+binWidth*xIdx;
			upper[0]=logxyMin+binWidth*(xIdx+1);
			for(unsigned int yIdx=0; yIdx<divisions; yIdx++){
				lower[1]=logxyMin+binWidth*yIdx;
				upper[1]=logxyMin+binWidth*(yIdx+1);
				//std::cout << "Integrating x [" << lower[0] << ',' << upper[0] 
				//<< "], y [" << lower[1] << ',' << upper[1] << ']' << std::endl;
				
				using StatePtr=std::unique_ptr<gsl_monte_vegas_state,void(*)(gsl_monte_vegas_state*)>;
				StatePtr state(gsl_monte_vegas_alloc(2),gsl_monte_vegas_free);
				gsl_monte_function integrand={(gslFuncType)gslWrapper, 2, &params};
				size_t calls = 1e3;
				bool lastIterationZero=false;
				double result, error;
				do{
					gsl_monte_vegas_integrate (&integrand, lower, upper, 2, calls, r.get(), state.get(), &result, &error);
					if(gsl_monte_vegas_chisq(state.get())==0){
						//two iterations in a row with chi^2 exactly zero probably 
						//indicates that we are integrating a constant function, so just stop
						if(lastIterationZero)
							break;
						lastIterationZero=true;
					}
				}while(std::abs(gsl_monte_vegas_chisq(state.get())-1)>.1);
				
				expectations[xIdx][yIdx]=result;
				total+=result;
			}
		}
		std::cout << "Total by section integration: " << total << std::endl;
		std::cout << "Total from lookup: " << xs.evaluateTotalCrossSection(energy) << std::endl;
		ENSURE_DISTANCE(total,xs.evaluateTotalCrossSection(energy),.01*total,
		                "Sum of partial integrated corss sections should equal the total cross section");
		//normalize to number of trials
		for(unsigned int xIdx=0; xIdx<divisions; xIdx++){
			for(unsigned int yIdx=0; yIdx<divisions; yIdx++){
				expectations[xIdx][yIdx]*=config.events/total;
			}
		}
	}
	
	//finally, compare the observed samples to the expectations
	{
		unsigned int nSuccess=0, nNonzero=0;
		for(unsigned int xIdx=0; xIdx<divisions; xIdx++){
			for(unsigned int yIdx=0; yIdx<divisions; yIdx++){
				std::cout << xIdx << ',' << yIdx << ": expect " << expectations[xIdx][yIdx]
					<< " observe " << counts[xIdx][yIdx] << std::endl;
				if(expectations[xIdx][yIdx]==0){
					if(counts[xIdx][yIdx]==0)
						nSuccess++;
				}
				else{
					nNonzero++;
					if(std::abs(double(counts[xIdx][yIdx])-expectations[xIdx][yIdx])<sqrt(expectations[xIdx][yIdx]))
						nSuccess++;
				}
			}
		}
		//we expect all zero bins to agree, and ~68% of nonzero bins to agree
		double expectedSuccesses=(divisions*divisions-nNonzero)+0.68*nNonzero;
		std::cout << "Success rate: " << nSuccess << '/' << (divisions*divisions) 
			<< '=' << double(nSuccess)/(divisions*divisions) << std::endl;
		std::cout << expectedSuccesses << " expected" << std::endl;
		ENSURE_DISTANCE(nSuccess,expectedSuccesses,.05*(divisions*divisions));
	}
}

TEST(9_impact_parameter_distribution){
	RangedInjectionConfiguration config;
	config.events=30000;
	{ //standard injection radius
		MomentAccumulator moments;
		double minImpactSeen=1e6, maxImpactSeen=-1;
		
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			
			//check that the stored impact parameter is correct by comparing to the MCTree
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			I3Position v=primary.GetPos();
			I3Direction d=primary.GetDir();
			double impactParameter=(v-d*(v*d)).Magnitude();
			ENSURE_DISTANCE(props.impactParameter, impactParameter, .001);
			
			moments.Insert(props.impactParameter);
			minImpactSeen=std::min(minImpactSeen,props.impactParameter);
			maxImpactSeen=std::max(maxImpactSeen,props.impactParameter);
		}
		
		//check the range
		ENSURE(minImpactSeen>=0,"Sampled impact parameters should be in bounds");
		ENSURE(maxImpactSeen<=config.injectionRadius,
			   "Sampled impact parameters should be in bounds");
		ENSURE_DISTANCE(minImpactSeen,0,20,
						"Minimum impact parameter should be approached");
		ENSURE_DISTANCE(maxImpactSeen,config.injectionRadius,1,
						"Maximum impact parameter should be approached");
		//check the shape
		//index is -1 because the distribution of radii should be proportional to radius
		testPowerLawness(-1, 0, config.injectionRadius, config.events,
						 moments, __FILE__, __LINE__);
	}
	
	{ //reduced injection radius
		config.injectionRadius=100;
		MomentAccumulator moments;
		double minImpactSeen=1e6, maxImpactSeen=-1;
		
		RangedLeptonInjector inj(context,config);
		ConfigureEnergyRange(inj,1e2,1e6);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			
			//check that the stored impact parameter is correct by comparing to the MCTree
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			I3Position v=primary.GetPos();
			I3Direction d=primary.GetDir();
			double impactParameter=(v-d*(v*d)).Magnitude();
			ENSURE_DISTANCE(props.impactParameter, impactParameter, .001);
			
			moments.Insert(props.impactParameter);
			minImpactSeen=std::min(minImpactSeen,props.impactParameter);
			maxImpactSeen=std::max(maxImpactSeen,props.impactParameter);
		}
		
		//check the range
		ENSURE(minImpactSeen>=0,"Sampled impact parameters should be in bounds");
		ENSURE(maxImpactSeen<=config.injectionRadius,
			   "Sampled impact parameters should be in bounds");
		ENSURE_DISTANCE(minImpactSeen,0,1,
						"Minimum impact parameter should be approached");
		ENSURE_DISTANCE(maxImpactSeen,config.injectionRadius,1,
						"Maximum impact parameter should be approached");
		//check the shape
		//index is -1 because the distribution of radii should be proportional to radius
		testPowerLawness(-1, 0, config.injectionRadius, config.events,
						 moments, __FILE__, __LINE__);
	}
}

TEST(A_column_depth_distribution){
	using namespace earthmodel::EarthModelCalculator;
	extern boost::shared_ptr<earthmodel::EarthModelService> earthmodelService;
	ENSURE((bool)earthmodelService);
	const double pi=4*atan(1.);
	
	RangedInjectionConfiguration config;
	config.events=10000;
	//fixed energy
	config.energyMinimum=1*I3Units::TeV;
	config.energyMaximum=1*I3Units::TeV;
	//fixed position
	config.injectionRadius=0;
	//NuMu CC
	config.finalType1=I3Particle::MuMinus;
	config.finalType2=I3Particle::Hadrons;
	
	{ //uniform material case
		//fixed (horizontal) direction
		config.azimuthMinimum=0;
		config.azimuthMaximum=0;
		config.zenithMinimum=pi/2;
		config.zenithMaximum=pi/2;
		
		MomentAccumulator positionMoments, columnDepthMoments;
		
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		
		I3Position center(0,0,0);
		I3Direction dir(config.zenithMinimum,config.azimuthMinimum);
		ENSURE_EQUAL(config.energyMinimum,config.energyMaximum,"This test must be mono-energetic");
		double totalColumnDepth=MWEtoColumnDepthCGS(GetLeptonRange(config.energyMinimum))
		+earthmodelService->GetColumnDepthInCGS(center-config.endcapLength*dir,center+config.endcapLength*dir);
		double expectedMaxDist=earthmodelService->DistanceForColumnDepthToPoint(center+config.endcapLength*dir,dir,totalColumnDepth)-config.endcapLength;
		std::cout << "Expected maximum distance: " << expectedMaxDist << std::endl;
		
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			
			ENSURE_EQUAL(props.totalColumnDepth,totalColumnDepth);
			positionMoments.Insert(primary.GetPos().GetX());
			columnDepthMoments.Insert(earthmodelService->GetColumnDepthInCGS(primary.GetPos(),center+config.endcapLength*dir));
		}
		
		testPowerLawness(0, -config.endcapLength, expectedMaxDist, config.events,
						 positionMoments, __FILE__, __LINE__);
		testPowerLawness(0, 0, totalColumnDepth, config.events,
						 columnDepthMoments, __FILE__, __LINE__);
	}
	
	{ //non-uniform material case
		//positions will not be uniformly distributed, but column depths should be
		
		//fixed (vetical) direction
		config.azimuthMinimum=0;
		config.azimuthMaximum=0;
		config.zenithMinimum=pi;
		config.zenithMaximum=pi;
		
		MomentAccumulator positionMoments, columnDepthMoments;
		
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		
		I3Position center(0,0,0);
		I3Direction dir(config.zenithMinimum,config.azimuthMinimum);
		ENSURE_EQUAL(config.energyMinimum,config.energyMaximum,"This test must be mono-energetic");
		double totalColumnDepth=MWEtoColumnDepthCGS(GetLeptonRange(config.energyMinimum))
		+earthmodelService->GetColumnDepthInCGS(center-config.endcapLength*dir,center+config.endcapLength*dir);
		std::cout << "Total column depth: " << totalColumnDepth << std::endl;
		double expectedMaxDist=earthmodelService->DistanceForColumnDepthToPoint(center+config.endcapLength*dir,dir,totalColumnDepth)-config.endcapLength;
		std::cout << "Expected maximum distance: " << expectedMaxDist << std::endl;
		
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			
			ENSURE_EQUAL(props.totalColumnDepth,totalColumnDepth);
			positionMoments.Insert(primary.GetPos().GetZ());
			columnDepthMoments.Insert(earthmodelService->GetColumnDepthInCGS(primary.GetPos(),center+config.endcapLength*dir));
		}
		
		//This would be the test of uniformity of postions, except that the positions
		//should not be uniform in this case.
		//testPowerLawness(0, -expectedMaxDist, config.endcapLength, config.events,
		//				 positionMoments, __FILE__, __LINE__);
		testPowerLawness(0, 0, totalColumnDepth, config.events,
						 columnDepthMoments, __FILE__, __LINE__);
	}

	{ //clipped case
		//above the detector there may not be enough column depth for the maximum muon range
		//not only should the positions be kept inside the atmosphere, but the total column
		//depths reported should be reduced to match the total amount of material present
		
		//fixed (vetical) direction
		config.azimuthMinimum=0;
		config.azimuthMaximum=0;
		config.zenithMinimum=0;
		config.zenithMaximum=0;
		
		MomentAccumulator positionMoments, columnDepthMoments;
		
		RangedLeptonInjector inj(context,config);
		ConfigureStandardParams(inj);
		boost::shared_ptr<OutputCollector> col=connectCollector(inj);
		
		I3Position center(0,0,0);
		I3Direction dir(config.zenithMinimum,config.azimuthMinimum);
		ENSURE_EQUAL(config.energyMinimum,config.energyMaximum,"This test must be mono-energetic");
		double totalColumnDepth=MWEtoColumnDepthCGS(GetLeptonRange(config.energyMinimum))
		+earthmodelService->GetColumnDepthInCGS(center-config.endcapLength*dir,center+config.endcapLength*dir);
		std::cout << "Total column depth: " << totalColumnDepth << std::endl;
		double expectedMaxDist=earthmodelService->DistanceForColumnDepthToPoint(center+config.endcapLength*dir,dir,totalColumnDepth)-config.endcapLength;
		std::cout << "Expected maximum distance: " << expectedMaxDist << std::endl;
		double actualColumnDepth=earthmodelService->GetColumnDepthInCGS(center+config.endcapLength*dir,center-expectedMaxDist*dir);
		std::cout << "Actually availiable column depth: " << actualColumnDepth << std::endl;
	
		while(!inj.DoneGenerating()){
			boost::shared_ptr<I3Frame> f(new I3Frame('Q'));
			inj.DAQ(f);
			const RangedEventProperties& props=f->Get<RangedEventProperties>("EventProperties");
			const I3MCTree& tree=f->Get<const I3MCTree>();
			ENSURE(!tree.empty(),"The MCTree should not be empty");
			std::vector<I3Particle> primaries=I3MCTreeUtils::GetPrimaries(tree);
			ENSURE(primaries.size()==1,"There should be one primary");
			I3Particle primary=primaries.front();
			
			double columnDepth=earthmodelService->GetColumnDepthInCGS(primary.GetPos(),center+config.endcapLength*dir);
			ENSURE_EQUAL(props.totalColumnDepth,actualColumnDepth);
			positionMoments.Insert(primary.GetPos().GetZ());
			columnDepthMoments.Insert(columnDepth);
		}
		
		//This would be the test of uniformity of postions, except that the positions
		//should not be uniform in this case.
		//testPowerLawness(0, -expectedMaxDist, config.endcapLength, config.events,
		//				 positionMoments, __FILE__, __LINE__);
		testPowerLawness(0, 0, actualColumnDepth, config.events,
						 columnDepthMoments, __FILE__, __LINE__);
	}
}
