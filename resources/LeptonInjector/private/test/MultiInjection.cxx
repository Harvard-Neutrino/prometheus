#include <I3Test.h>

#include <fstream>

#include <icetray/I3Tray.h>
#include <dataclasses/physics/I3MCTreeUtils.h>
#include <LeptonInjector/LeptonInjector.h>

#include "tools.h"

TEST_GROUP(MultiInjection);

using namespace LeptonInjector;

//None of the default values in the *Configuration objects should set off
//errors in the module
TEST(1_sane_defaults_in_config){
	std::cout << "creating rconfig" << std::endl;
	RangedInjectionConfiguration rconfig;
	std::cout << "creating vconfig" << std::endl;
	VolumeInjectionConfiguration vconfig;
	std::cout << "constructing MultiLeptonInjector" << std::endl;
	MultiLeptonInjector inj(context,rconfig,vconfig);
	std::cout << "calling ConfigureStandardParams" << std::endl;
	ConfigureStandardParams(inj);
}

//attempting to set nonsensical parameter values should be rejected
TEST(2_reject_invalid_params){
	RangedInjectionConfiguration rconfig;
	VolumeInjectionConfiguration vconfig;
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.energyMinimum=0;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a non-positive minimum energy should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.energyMaximum=0;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a non-positive maximum energy should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.energyMinimum=20;
		rconfig.energyMaximum=10;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum energy below the minimum energy should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.azimuthMinimum=-2;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative minimum azimuth should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.azimuthMaximum=7;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum azimuth greater than 2 pi should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.azimuthMinimum=2;
		rconfig.azimuthMaximum=1;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum azimuth less than the minimum azimuth should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.zenithMinimum=-2;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative minimum zenith should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.zenithMaximum=4;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum zenith greater than pi should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.zenithMinimum=2;
		rconfig.zenithMaximum=1;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a maximum zenith less than the minimum zenith should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		I3Context context;
		//no random service
		context.Put(earthmodelService,earthModelName);
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("not finding the random service in the cntext should be detected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		I3Context context;
		context.Put(randomService);
		//no earth model service
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("not finding the Earth model service in the cntext should be detected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		//don't set cross section file
		inj.GetConfiguration().Set("EarthModel",boost::python::object(earthModelName));
		inj.Configure();
		FAIL("configuring without a cross section should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		//don't set earth model service
		inj.Configure();
		FAIL("configuring without an earth model service should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.injectionRadius=-200;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative injection radius should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		RangedInjectionConfiguration rconfig;
		rconfig.endcapLength=-200;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative endcap length should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		VolumeInjectionConfiguration vconfig;
		vconfig.cylinderRadius=-200;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative cylinder radius should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
	
	try{
		VolumeInjectionConfiguration vconfig;
		vconfig.cylinderHeight=-200;
		MultiLeptonInjector inj(context,rconfig,vconfig);
		ConfigureStandardParams(inj);
		FAIL("configuring with a negative cylinder height should be rejected");
	}catch(std::runtime_error& e){/*squash*/}
}

TEST(3_number_of_events){
	const unsigned int nEvents=100;
	const std::string filename=I3Test::testfile("Multi_NEvents_test.i3");
	
	std::vector<Injector> generatorSettings;
	generatorSettings.push_back(Injector(nEvents,I3Particle::MuMinus,I3Particle::Hadrons,defaultCrosssectionPath,defaultTotalCrosssectionPath,true)); //CC
	generatorSettings.push_back(Injector(nEvents,I3Particle::NuMu,I3Particle::Hadrons,defaultCrosssectionPath,defaultTotalCrosssectionPath,false)); //NC
	
	I3Tray tray;
	tray.GetContext().Put(randomService);
	tray.GetContext().Put(earthmodelService,earthModelName);
	tray.AddModule("I3InfiniteSource")("Stream",I3Frame::Stream('Q'));
	tray.AddModule("MultiLeptonInjector")
	("Generators",generatorSettings)
	("MinimumEnergy",1e2)
	("MaximumEnergy",1e6)
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
	
	for(unsigned int j=0; j<generatorSettings.size(); j++){
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
	}
	
	//There should be no more frames
	f.load(resultsFile);
	ENSURE(resultsFile.fail());
	
	//If the test was successful, clean up
	unlink(filename.c_str());
}
