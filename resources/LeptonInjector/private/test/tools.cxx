#include "tools.h"

#include <boost/python/import.hpp>

#include <I3Test.h>
#include <icetray/init.h>
#include <phys-services/I3GSLRandomService.h>

bool initialized=false;
I3Context context;
const std::string earthModelName="Earth";
//TODO: should put test cross sections in test data and fetch from I3_TESTDATA
const std::string defaultCrosssectionPath=std::string(getenv("I3_SRC"))+"/LeptonInjector/resources/test_xs.fits";
const std::string defaultTotalCrosssectionPath=std::string(getenv("I3_SRC"))+"/LeptonInjector/resources/test_xs_total.fits";

boost::shared_ptr<I3RandomService> randomService;
boost::shared_ptr<earthmodel::EarthModelService> earthmodelService;
boost::shared_ptr<I3FrameObject> initialRandomState;

struct service_initializer{
	service_initializer(){
		if(initialized)
			return;
		initialized=true;
		I3::init_icetray_lib();
		boost::python::import("icecube.dataclasses");
		try{
			boost::python::import("icecube.LeptonInjector");
		}catch(boost::python::error_already_set& eas){
			PyErr_Print();
			log_fatal("Blerg, bad.");
		}
		GetIcetrayLogger()->SetLogLevel(I3LOG_ERROR);
		randomService=boost::make_shared<I3GSLRandomService>(128);
		initialRandomState=randomService->GetState();
		context.Put(randomService);
		earthmodelService=boost::make_shared<earthmodel::EarthModelService>();
		context.Put(earthmodelService,earthModelName);
	}
} inits;

void resetRandomState(){
	randomService->RestoreState(initialRandomState);
}

void ConfigureStandardParams(I3Module& inj){
	if(inj.GetConfiguration().Has("DoublyDifferentialCrossSectionFile"))
		inj.GetConfiguration().Set("DoublyDifferentialCrossSectionFile",boost::python::object(defaultCrosssectionPath));
	if(inj.GetConfiguration().Has("TotalCrossSectionFile"))
		inj.GetConfiguration().Set("TotalCrossSectionFile",boost::python::object(defaultTotalCrosssectionPath));
	if(inj.GetConfiguration().Has("EarthModel")) //RangedInjector
		inj.GetConfiguration().Set("EarthModel",boost::python::object(earthModelName));
	if(inj.GetConfiguration().Has("Generators")){ //MultiInjector
		using LeptonInjector::Injector;
		std::vector<Injector> generatorSettings;
		generatorSettings.push_back(Injector(1,I3Particle::MuMinus,I3Particle::Hadrons,defaultCrosssectionPath,defaultTotalCrosssectionPath,true)); //CC
		generatorSettings.push_back(Injector(1,I3Particle::NuMu,I3Particle::Hadrons,defaultCrosssectionPath,defaultTotalCrosssectionPath,false)); //NC
		inj.GetConfiguration().Set("Generators",boost::python::object(generatorSettings));
	}
	if(inj.GetConfiguration().Has("SuspendOnCompletion"))
		inj.GetConfiguration().Set("SuspendOnCompletion",boost::python::object(false));
	//inj.GetConfiguration().Set("MinimumEnergy",boost::python::object(1e2));
	//inj.GetConfiguration().Set("MaximumEnergy",boost::python::object(1e6));
	inj.Configure();
}

void ConfigureEnergyRange(I3Module& inj, double minEnergy, double maxEnergy){
	inj.GetConfiguration().Set("MinimumEnergy",boost::python::object(minEnergy));
	inj.GetConfiguration().Set("MaximumEnergy",boost::python::object(maxEnergy));
}

struct powerLawMomentIntegrand{
	double index, moment, norm, offset;
	powerLawMomentIntegrand(double i, unsigned int m, double n, double o=0):
	index(i),moment(m),norm(n),offset(o){}
	double operator()(double x) const{
		return(pow(x-offset,moment)*pow(x,-index)/norm);
	}
};

double predictPowerLawMoment(double index, double a, double b, unsigned int moment){
	using namespace earthmodel::Integration;
	double norm=rombergIntegrate(powerLawMomentIntegrand(index,0,1.),a,b);
	double offset=0;
	if(moment>1){
		//for moments above the mean, compute the central moment by using the mean as an offset
		offset=rombergIntegrate(powerLawMomentIntegrand(index,1,norm),a,b);
	}
	return(rombergIntegrate(powerLawMomentIntegrand(index,moment,norm,offset),a,b,1e-12));
}

void testPowerLawness(double powerlawIndex, double min, double max, unsigned int count,
					  const MomentAccumulator& moments, const std::string& file, unsigned line){
	double expectedMean=predictPowerLawMoment(powerlawIndex,min,max,1);
	double expectedVariance=predictPowerLawMoment(powerlawIndex,min,max,2);
	double cm3=0;
	if(powerlawIndex!=0)
		cm3=predictPowerLawMoment(powerlawIndex,min,max,3);
	double expectedSkewness=cm3/pow(expectedVariance,1.5);
	double cm4=predictPowerLawMoment(powerlawIndex,min,max,4);
	double expectedKurtosis=cm4/(expectedVariance*expectedVariance) - 3;
	if(powerlawIndex!=0)
		std::cout << "Powerlaw: E^" << (-powerlawIndex);
	else
		std::cout << "Uniform";
	std::cout << " from " << min << " to " << max << std::endl;
	std::cout << "Expected moments: " << expectedMean << ' ' << expectedVariance << ' ' << expectedSkewness << ' ' << expectedKurtosis << std::endl;
	std::cout << "Observed moments: " << moments.Mean() << ' ' << moments.Variance() << ' ' << moments.Skewness() << ' ' << moments.Kurtosis() << std::endl;
	
	//make failures look like they came from our caller
#define ENSURE_DISTANCE_FWD(LEFT,RIGHT,DISTANCE,...)     \
	I3Test::ensure_distance(file,line,                   \
	BOOST_PP_STRINGIZE(LEFT), BOOST_PP_STRINGIZE(RIGHT), \
	BOOST_PP_STRINGIZE(DISTANCE), LEFT,RIGHT,DISTANCE,	 \
	##__VA_ARGS__);
	
	ENSURE_DISTANCE_FWD(moments.Mean(),expectedMean,
						3*sqrt(expectedVariance/count),
						"Should have the correct mean");//P(fail)=~2.7e-3
	ENSURE_DISTANCE_FWD(moments.Variance(),expectedVariance,.03*expectedVariance,
						"Should have the correct variance");
	ENSURE_DISTANCE_FWD(moments.Skewness(),expectedSkewness,std::max(.05,.05*std::abs(expectedSkewness)),
						"Should have the correct skewness");
	ENSURE_DISTANCE_FWD(moments.Kurtosis(),expectedKurtosis,std::max(.05,.05*std::abs(expectedKurtosis)),
						"Should have the correct kurtosis");
#undef ENSURE_DISTANCE_FWD
}

boost::shared_ptr<LeptonInjector::OutputCollector> connectCollector(I3Module& mod){
	using LeptonInjector::OutputCollector;
	boost::shared_ptr<OutputCollector> col=boost::make_shared<OutputCollector>(mod.GetContext());
	mod.ConnectOutBox("OutBox",col);
	return(col);
}
