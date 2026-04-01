#include <vector>
#include <string>
#include <map>
#include <utility>
#include <tuple>

#include <LeptonInjector/LeptonInjector.h>
#include <LeptonInjector/Coordinates.h>
#include <LeptonInjector/Controller.h>
#include <LeptonInjector/Random.h>
#include <LeptonInjector/Constants.h>
#include <earthmodel-service/EarthModelCalculator.h>
#include <earthmodel-service/EarthModelService.h>

// #include <converter/LeptonInjectionConfigurationConverter.h>
#include <boost/python.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_value_policy.hpp>


#include "container_conversions.h"

using namespace boost::python;

template<class T>
struct VecToList
{
	static PyObject* convert(const std::vector<T>& vec){
		boost::python::list* l = new boost::python::list();
		for(size_t i =0; i < vec.size(); i++)
			(*l).append(vec[i]);

		return l->ptr();
	}
};


template<typename T, typename U, typename P>
struct ThreeTupleToPyTuple
{
	static PyObject* convert(const std::tuple<T,U,P>& tup){
		boost::python::list* l = new boost::python::list();
		(*l).append(std::get<0>(tup));
		(*l).append(std::get<1>(tup));
		(*l).append(std::get<2>(tup));

		return l->ptr();
	}
};

template<class T>
struct DeqToList
{
	static PyObject* convert(const std::deque<T>& vec){
		boost::python::list* l = new boost::python::list();
		for(size_t i =0; i < vec.size(); i++)
			(*l).append(vec[i]);

		return l->ptr();
	}
};

void ListToVec(std::vector<unsigned int> &ret, boost::python::list l){
	for(int i=0;i<boost::python::len(l);i++)
		ret.push_back(boost::python::extract<unsigned int>(l[i]));
}

namespace earthmodel{
	struct LIEarthModelCalculator {
	};
}

BOOST_PYTHON_MODULE(EarthModelService){
	using namespace earthmodel;

	enum_<EarthModelService::MediumType>("MediumType")
		.value("INNERCORE", EarthModelService::INNERCORE)
		.value("OUTERCORE", EarthModelService::OUTERCORE)
		.value("MANTLE", EarthModelService::MANTLE)
		.value("ROCK", EarthModelService::ROCK)
		.value("ICE", EarthModelService::ICE)
		.value("WATER", EarthModelService::WATER)
		.value("AIR", EarthModelService::AIR)
		.value("VACUUM", EarthModelService::VACUUM)
		.value("LOWERMANTLE", EarthModelService::LOWERMANTLE)
		.value("UPPERMANTLE", EarthModelService::UPPERMANTLE)
		.value("LLSVP", EarthModelService::LLSVP)
		;

	class_<EarthModelService::EarthParam>("EarthParam", init<>())
		.def_readwrite("fUpperRadius_",&EarthModelService::EarthParam::fUpperRadius_)
		.def_readwrite("fZOffset_",&EarthModelService::EarthParam::fZOffset_)
		.def_readwrite("fBoundaryName_",&EarthModelService::EarthParam::fBoundaryName_)
		.def_readwrite("fMediumType_",&EarthModelService::EarthParam::fMediumType_)
		.def_readwrite("fParams_",&EarthModelService::EarthParam::fParams_)
		.def("GetDensity",&EarthModelService::EarthParam::GetDensity)
		.def("PrintDensity",&EarthModelService::EarthParam::PrintDensity)
		;

	const double (EarthModelService::*GetEarthDensityInCGS)(const LeptonInjector::LI_Position&) const = &EarthModelService::GetEarthDensityInCGS;
	double (*GetEarthLayerDensityInCGS)(const EarthModelService::EarthParam&, const LeptonInjector::LI_Position&) = &EarthModelService::GetEarthDensityInCGS;

	const double (EarthModelService::*GetDensityInCGS)(const LeptonInjector::LI_Position&) const = &EarthModelService::GetDensityInCGS;
	const double (EarthModelService::*GetLayerDensityInCGS)(const EarthModelService::EarthParam&, const LeptonInjector::LI_Position&) = &EarthModelService::GetDensityInCGS;


	const EarthModelService::MatRatioMap (EarthModelService::*GetMatRatioMap)() const = &EarthModelService::GetMatRatioMap;
	const std::map<int, double>& (EarthModelService::*GetMediumMatRatioMap)(EarthModelService::MediumType) const = &EarthModelService::GetMatRatioMap;

	class_<EarthModelService>("EarthModelService", init<const std::string&, const std::string&, const std::vector<std::string>&, const std::vector<std::string>&, const std::string&, double, double>(
        (args("name"), args("table path"), args("earth models"),args("material models"), args("ice cap name"), args("ice cap angle"), args("detector depth")))
    )
		// .def("GetMediumTypeString",&EarthModelService::GetMediumTypeString)
		// .def("ConvertMediumTypeString",&EarthModelService::ConvertMediumTypeString)
		.def("GetEarthParam",&EarthModelService::GetEarthParam, return_value_policy<copy_const_reference>())
		.def("GetEarthDensityInCGS",GetEarthDensityInCGS)
		.def("GetEarthLayerDensityInCGS",GetEarthLayerDensityInCGS)
		.def("GetDensityInCGS",GetDensityInCGS)
		.def("GetLayerDensityInCGS",GetLayerDensityInCGS)
		.def("GetColumnDepthInCGS",&EarthModelService::GetColumnDepthInCGS)
        .def("GetEarthDensitySegments",&EarthModelService::GetEarthDensitySegments)
        .def("GetDensitySegments",&EarthModelService::GetDensitySegments)
		// .def("IntegrateDensityInCGS",&EarthModelService::IntegrateDensityInCGS)
		.def("DistanceForColumnDepthToPoint",&EarthModelService::DistanceForColumnDepthToPoint)
		// .def("DistanceForColumnDepthFromPoint",&EarthModelService::DistanceForColumnDepthFromPoint)
		// .def("GetLeptonRangeInMeterFrom",&EarthModelService::GetLeptonRangeInMeterFrom)
		// .def("GetLeptonRangeInMeterTo",&EarthModelService::GetLeptonRangeInMeterTo)
		// .def("DistanceToNextBoundaryCrossing",&EarthModelService::DistanceToNextBoundaryCrossing)
		// .def("GetMedium",&EarthModelService::GetMedium)
		//.def("GetMatRatioMap",GetMatRatioMap)
		//.def("GetMediumMatRatioMap",GetMediumMatRatioMap)
		// .def("GetMatRatio",&EarthModelService::GetMatRatio)
		//.def("GetPNERatioMap",&EarthModelService::GetPNERatioMap)
		.def("GetPNERatio",&EarthModelService::GetPNERatio)
		// .def("GetDistanceFromEarthEntranceToDetector",&EarthModelService::GetDistanceFromEarthEntranceToDetector)
		// .def("GetDistanceFromSphereSurfaceToDetector",&EarthModelService::GetDistanceFromSphereSurfaceToDetector)
		// .def("PrintEarthParams",&EarthModelService::PrintEarthParams)
		// .def("GetPREM",&EarthModelService::GetPREM)
		.def("GetEarthCoordPosFromDetCoordPos",&EarthModelService::GetEarthCoordPosFromDetCoordPos)
		.def("GetDetCoordPosFromEarthCoordPos",&EarthModelService::GetDetCoordPosFromEarthCoordPos)
		.def("GetEarthCoordDirFromDetCoordDir",&EarthModelService::GetEarthCoordDirFromDetCoordDir)
		// .def("GetDetCoordDirFromEarthCoordDir",&EarthModelService::GetDetCoordDirFromEarthCoordDir)
		// .def("SetPath",&EarthModelService::SetPath)
		// .def("SetEarthModel",&EarthModelService::SetEarthModel)
		// .def("SetMaterialModel",&EarthModelService::SetMaterialModel)
		// .def("SetIceCapTypeString",&EarthModelService::SetIceCapTypeString)
		// .def("SetIceCapSimpleAngle",&EarthModelService::SetIceCapSimpleAngle)
		// .def("SetDetectorDepth",&EarthModelService::SetDetectorDepth)
		// .def("SetDetectorXY",&EarthModelService::SetDetectorXY)
		// .def("GetDetectorDepth",&EarthModelService::GetDetectorDepth)
		// .def("GetDetectorPosInEarthCoord",&EarthModelService::GetDetectorPosInEarthCoord)
		// .def("GetPath",&EarthModelService::GetPath)
		// .def("GetIceCapTypeString",&EarthModelService::GetIceCapTypeString)
		// .def("GetIceCapSimpleAngle",&EarthModelService::GetIceCapSimpleAngle)
		// .def("GetBoundary",&EarthModelService::GetBoundary)
		// .def("GetMohoBoundary",&EarthModelService::GetMohoBoundary)
		// .def("GetRockIceBoundary",&EarthModelService::GetRockIceBoundary)
		// .def("GetIceAirBoundary",&EarthModelService::GetIceAirBoundary)
		.def("GetAtmoRadius",&EarthModelService::GetAtmoRadius)
		// .def("RadiusToCosZen",&EarthModelService::RadiusToCosZen)
		// .def("Init",&EarthModelService::Init)
		;

	{
		scope earthmodel = class_<LIEarthModelCalculator>("EarthModelCalculator");

		enum_<EarthModelCalculator::LeptonRangeOption>("LeptonRangeOption")
			.value("DEFAULT", EarthModelCalculator::DEFAULT)
			.value("LEGACY", EarthModelCalculator::LEGACY)
			.value("NUSIM", EarthModelCalculator::NUSIM)
			;

		def("GetImpactParameter", &EarthModelCalculator::GetImpactParameter);
		def("GetIntersectionsWithSphere", &EarthModelCalculator::GetIntersectionsWithSphere);
		def("GetDistsToIntersectionsWithSphere", &EarthModelCalculator::GetDistsToIntersectionsWithSphere);
		def("GetLeptonRange", &EarthModelCalculator::GetLeptonRange);
		def("ColumnDepthCGStoMWE",&EarthModelCalculator::ColumnDepthCGStoMWE);
		def("MWEtoColumnDepthCGS",&EarthModelCalculator::MWEtoColumnDepthCGS);
	}

	{
		scope integration = class_<LIEarthModelCalculator>("Integration");
	}

	using namespace scitbx::boost_python::container_conversions;
	from_python_sequence< std::vector<double>, variable_capacity_policy >();
	to_python_converter< std::vector<double, class std::allocator<double> >, VecToList<double> > ();

    //from_python_sequence< std::tuple<double,double,double>, variable_capacity_policy >();
	to_python_converter< std::tuple<double, double,double>, ThreeTupleToPyTuple<double,double,double> > ();

	from_python_sequence< std::vector<std::tuple<double,double,double>>, variable_capacity_policy >();
	to_python_converter< std::vector<std::tuple<double,double,double>, class std::allocator<std::tuple<double,double,double>> >, VecToList<std::tuple<double,double,double>> > ();

	from_python_sequence< std::vector<int>, variable_capacity_policy >();
	to_python_converter< std::vector<int, class std::allocator<int> >, VecToList<int> > ();

	from_python_sequence< std::vector<unsigned int>, variable_capacity_policy >();
	to_python_converter< std::vector<unsigned int, class std::allocator<unsigned int> >, VecToList<unsigned int> > ();

	from_python_sequence< std::vector<std::string>, variable_capacity_policy >();
	to_python_converter< std::vector<std::string, class std::allocator<std::string> >, VecToList<std::string> > ();
}
