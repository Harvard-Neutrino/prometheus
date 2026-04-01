#include <LeptonInjector/DataWriter.h>
#include <boost/version.hpp>
#include <boost/preprocessor/stringize.hpp>>
// From the IceCube tableio project 
#define WRAP_ELEMENTS(R,DATA,ELEM) BOOST_PP_STRINGIZE(ELEM),DATA::ELEM
#define MAKE_ENUM_VECTOR(VECTORNAME,CLASS,ENUM_TYPE,FIELDS)     \
    std::vector< std::pair<std::string,CLASS::ENUM_TYPE> > VECTORNAME ; \
    boost::assign::push_back(VECTORNAME) BOOST_PP_SEQ_TRANSFORM(WRAP_ELEMENTS,CLASS,FIELDS); \


namespace LeptonInjector{

// Commence the magic... Written by Chris Weaver for IceCube 

template<typename T>
struct endian_adapter{
    const T& t;
    endian_adapter(const T& t):t(t){}
};
	
template<typename T>
endian_adapter<T> little_endian(const T& t){ return(endian_adapter<T>(t)); }

std::ostream& endianWrite(std::ostream& os, const char* data, size_t dataSize){
#if BYTE_ORDER == LITTLE_ENDIAN
    //just write bytes
    os.write(data,dataSize);
#elif BYTE_ORDER == BIG_ENDIAN
    //write bytes in reverse order
    for(size_t i=1; i<=dataSize; i++)
        os.write(data+dataSize-i),1);
#elif BYTE_ORDER == PDP_ENDIAN
    //complain bitterly
    #error PDP-endian systems are not supported.
#else
    #error Unable to determine machine endianness!
#endif
    return(os);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const endian_adapter<T>& e){
    return(endianWrite(os,(char*)&e.t,sizeof(T)));
}

//automatically write string's length before its data
std::ostream& operator<<(std::ostream& os, const endian_adapter<std::string>& e){
    os << little_endian(e.t.size());
    return(endianWrite(os,e.t.c_str(),e.t.size()));
}

//same for vector<char>
std::ostream& operator<<(std::ostream& os, const endian_adapter<std::vector<char> >& e){
    os << little_endian(e.t.size());
    return(endianWrite(os,&e.t[0],e.t.size()));
}


/*
Base:
uint64_t: block length
size_t: name length
char[name length]: block type name
uint8_t: version number
char[block length-17-name length]: block data
*/
void writeBlockHeader(std::ostream& os, uint64_t blockDataSize,
                        const std::string& blockTypeName, uint8_t version){
    size_t nameLen=blockTypeName.size();
    uint64_t totalBlockSize=8+sizeof(nameLen)+nameLen+1+blockDataSize;
    os << little_endian(totalBlockSize)
        << little_endian(blockTypeName)
        << little_endian(version);
    if(!os.good()){
        std::cout << "Writing block header failed" << std::endl;
        throw;
    }
}

template<typename Enum>
void writeEnumDefBlock(std::ostream& os, const std::string& enumName,
                        const std::vector<std::pair<std::string,Enum> >& enumerators){
    //compute data size
    uint64_t dataSize=0;
    size_t nameLen=enumName.size();
    dataSize+=sizeof(nameLen);
    dataSize+=nameLen;
    if(enumerators.size()>=(1ULL<<32)){
        std::cout << "Number of enumerators (" << enumerators.size()
                            << ") too large" << std::endl;
        throw;
    }
    uint32_t numEnum=enumerators.size();
    dataSize+=sizeof(numEnum);
    for(size_t i=0; i<numEnum; i++){
        dataSize+=8+sizeof(size_t)+enumerators[i].first.size();
    }
    //write header
    writeBlockHeader(os,dataSize,"EnumDef",1);
    //write data
    os << little_endian(enumName) << little_endian(numEnum);
    for(size_t i=0; i<numEnum; i++){
        os << little_endian((int64_t)enumerators[i].second)
            << little_endian(enumerators[i].first);
    }
    if(!os.good()){
        std::cout << "bad problem writing enum block" << std::endl;
        throw;
    }
}

// End Magic 

DataWriter::DataWriter(){
}

// close all the hdf5 things that are open! 
DataWriter::~DataWriter(){
    //avoid the awkwatd situation where the deconstructot is called before you even do anything
    if(this->opened){
        herr_t status;
        status = H5Tclose( particleTable );
        status = H5Tclose( rangedPropertiesTable );
        status = H5Tclose( volumePropertiesTable );

        status = H5Gclose( group_handle ); 
        status = H5Dclose( initials );
        status = H5Dclose( final_1 );
        status = H5Dclose( final_2 );
        status = H5Dclose( properties );

        status = H5Fclose( fileHandle );
    }
    // deconstructor
    // should kill all the hdf5 stuff
}

// open the file, establish the datastructure, and create the datatypes that will be written
void DataWriter::OpenFile( std::string filename ){
    this->opened = true;
    // open a new file. If one already exists by the same name - overwrite it (H5F_ACC_TRUNC)
    // leave the defaults for property lists
    // the last two can be adjusted to allow for parallel file access as I understand, which may allow for multi-processing
    //      the different injectors, and therefore much quicker injection 
    fileHandle = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    makeTables();

    event_count = 0;

    if (fileHandle < 0 ){
        std::cout << "Unable to create file! Error no " << fileHandle << std::endl;
        throw;
    }

}

bool does_file_exist(const std::string filename){
    std::ifstream infile(filename);
    return infile.good();
}

void DataWriter::OpenLICFile( std::string filename ){
    /*
    This opens up a binary final to write the LeptonInjector Configuration. 
    */
    bool skip_enum = false;
    if (does_file_exist( filename.c_str())){
        if(!this->overwrite){
            skip_enum = true;
            std::cout<< "Note: LIC file already exists, appending to end." <<std::endl;
        }else{
            std::cout<< "Note: LIC file already exists, overwriting."<<std::endl;
        }
    }

    if(this->overwrite){
        this->lic_file_output.open( filename, std::ofstream::out | std::ofstream::trunc );
    }else{
        this->lic_file_output.open( filename, std::ofstream::out | std::ofstream::app );
    }
    if(!lic_file_output.good()){
	    std::cout << "Failed to open " << filename << " for writing LIC file" << std::endl;
        throw;
    }

    MAKE_ENUM_VECTOR(type,Particle,Particle::ParticleType,PARTICLE_H_Particle_ParticleType);
    if (!skip_enum){
        writeEnumDefBlock(lic_file_output, "Particle::ParticleType", type);
    }
}


void DataWriter::AddInjector( std::string injector_name , bool ranged){
    herr_t status;

    event_count = 0;

    if(name_iterator>0){
        status = H5Dclose(initials);
        status = H5Dclose(final_1);
        status = H5Dclose(final_2);
        status = H5Dclose(properties);
        status = H5Gclose( group_handle );
    }
    
    group_handle = H5Gcreate( fileHandle, (injector_name + std::to_string(name_iterator)).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    name_iterator++;

    // prepare the event properties writing dataspace! 
    const hsize_t ndims = 1;
    hsize_t dims[ndims]={0}; // current dimensionality of the file space is zero
    hsize_t max_dims[ndims] = {H5S_UNLIMITED}; // but can be expanded infinitely 
    hsize_t file_space = H5Screate_simple( ndims, dims, max_dims);

    hid_t plist = H5Pcreate( H5P_DATASET_CREATE);
    H5Pset_layout( plist, H5D_CHUNKED);
    // 32768
    // any time you have an infinite sized dimension, you need to chunk it. 1000 is a nice round number 
    const int nice_round_number = 32768;
    hsize_t chunk_dims[ndims] = {nice_round_number};
    H5Pset_chunk(plist, ndims, chunk_dims);

    const char* init_name = "initial";
    const char* final_1_name = "final_1";
    const char* final_2_name = "final_2";
    initials  = H5Dcreate(group_handle,init_name, particleTable, file_space, H5P_DEFAULT, plist, H5P_DEFAULT );
    final_1 = H5Dcreate(group_handle,final_1_name, particleTable, file_space, H5P_DEFAULT, plist, H5P_DEFAULT); 
    final_2 = H5Dcreate(group_handle,final_2_name, particleTable, file_space, H5P_DEFAULT, plist, H5P_DEFAULT); 
    H5Sclose(file_space);
    H5Pclose(plist);

    const hsize_t ndims2 =1; 
    hsize_t dims2[ndims2] = {0};
    hsize_t max_dims2[ndims2] = {H5S_UNLIMITED};
    hsize_t file_space2 = H5Screate_simple( ndims2, dims2, max_dims2);

    hid_t plist2 = H5Pcreate( H5P_DATASET_CREATE);
    H5Pset_layout( plist2, H5D_CHUNKED);
    hsize_t chunk_dims2[ndims2] = {nice_round_number};
    H5Pset_chunk(plist2, ndims2, chunk_dims2);

    const char* props = "properties";
    if (ranged){
        write_ranged = true;
        properties = H5Dcreate(group_handle, props , rangedPropertiesTable, file_space2, H5P_DEFAULT, plist2, H5P_DEFAULT );
    }else{
        write_ranged = false;
        properties = H5Dcreate(group_handle, props , volumePropertiesTable, file_space2, H5P_DEFAULT, plist2, H5P_DEFAULT );
    }

    H5Sclose(file_space2);
    H5Pclose(plist2);
    
}

void DataWriter::WriteConfig( BasicInjectionConfiguration& config , bool ranged){
    if (ranged) {
        this->writeRangedConfig( config );
    }else{
        this->writeVolumeConfig( config );
    }
}

void DataWriter::writeRangedConfig( BasicInjectionConfiguration& config ){
    //compute data size
    uint64_t dataSize=0;
    dataSize+=4; //events
    dataSize+=8; //energyMinimum
    dataSize+=8; //energyMaximum
    dataSize+=8; //powerlawIndex
    dataSize+=8; //azimuthMinimum
    dataSize+=8; //azimuthMaximum
    dataSize+=8; //zenithMinimum
    dataSize+=8; //zenithMaximum
    dataSize+=sizeof(Particle::ParticleType); //finalType1
    dataSize+=sizeof(Particle::ParticleType); //finalType2
    dataSize+=sizeof(size_t); //crossSection size
    dataSize+=config.crossSectionBlob.size(); //crossSection
    dataSize+=sizeof(size_t);
    dataSize+=config.totalCrossSectionBlob.size(); //crossSection
    dataSize+=8; //injectionRadius
    dataSize+=8; //endcapLength
    //write header
    writeBlockHeader(this->lic_file_output,dataSize,"RangedInjectionConfiguration",1);
    //write data
    this->lic_file_output << little_endian(config.events)
    << little_endian(config.energyMinimum/Constants::GeV)
    << little_endian(config.energyMaximum/Constants::GeV)
    << little_endian(config.powerlawIndex)
    << little_endian(config.azimuthMinimum/Constants::radian)
    << little_endian(config.azimuthMaximum/Constants::radian)
    << little_endian(config.zenithMinimum/Constants::radian)
    << little_endian(config.zenithMaximum/Constants::radian)
    << little_endian(config.finalType1)
    << little_endian(config.finalType2)
    << little_endian(config.crossSectionBlob)
    << little_endian(config.totalCrossSectionBlob)
    << little_endian(config.injectionRadius/Constants::meter)
    << little_endian(config.endcapLength/Constants::meter);
    if(!this->lic_file_output.good()){
        std::cout << "Writing ranged injection config block failed" << std::endl;
        throw;
    }
}

void DataWriter::writeVolumeConfig( BasicInjectionConfiguration& config ){
    //compute data size
    uint64_t dataSize=0;
    dataSize+=4; //events
    dataSize+=8; //energyMinimum
    dataSize+=8; //energyMaximum
    dataSize+=8; //powerlawIndex
    dataSize+=8; //azimuthMinimum
    dataSize+=8; //azimuthMaximum
    dataSize+=8; //zenithMinimum
    dataSize+=8; //zenithMaximum
    dataSize+=sizeof(Particle::ParticleType); //finalType1
    dataSize+=sizeof(Particle::ParticleType); //finalType2
    dataSize+=sizeof(size_t); //crossSection size
    dataSize+=config.crossSectionBlob.size(); //crossSection
    dataSize+=sizeof(size_t);
    dataSize+=config.totalCrossSectionBlob.size(); //crossSection
    dataSize+=8; //cylinderRadius
    dataSize+=8; //cylinderHeight
    //write header
    writeBlockHeader(this->lic_file_output,dataSize,"VolumeInjectionConfiguration",1);
    //write data
    this->lic_file_output << little_endian(config.events)
        << little_endian(config.energyMinimum/Constants::GeV)
        << little_endian(config.energyMaximum/Constants::GeV)
        << little_endian(config.powerlawIndex)
        << little_endian(config.azimuthMinimum/Constants::radian)
        << little_endian(config.azimuthMaximum/Constants::radian)
        << little_endian(config.zenithMinimum/Constants::radian)
        << little_endian(config.zenithMaximum/Constants::radian)
        << little_endian(config.finalType1)
        << little_endian(config.finalType2)
        << little_endian(config.crossSectionBlob)
        << little_endian(config.totalCrossSectionBlob)
        << little_endian(config.cylinderRadius/Constants::meter)
        << little_endian(config.cylinderHeight/Constants::meter);
    if(!this->lic_file_output.good()){
        std::cout << "Writing volume injection config block failed" << std::endl;
        throw;
    }
}


void DataWriter::WriteEvent( BasicEventProperties& props, h5Particle& part1, h5Particle& part2, h5Particle& part3 ){
    // std::cout << " writing an event" << std::endl;
    hid_t memspace, file_space;
    const hsize_t n_dims = 1;
    hsize_t dims[n_dims] = {1};
    memspace = H5Screate_simple(n_dims, dims, NULL);

    //Extend dataset
    dims[0] = event_count+1;
    H5Dset_extent(initials, dims);
    H5Dset_extent(final_1, dims);
    H5Dset_extent(final_2, dims);
    H5Dset_extent(properties, dims);

    //Write waveforms
    file_space = H5Dget_space(initials);
    hsize_t start[n_dims] = {event_count};
    hsize_t count[n_dims] = {1};
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);


    // h5Particle temp_data[3] = {part1, part2, part3};
    H5Dwrite(initials, particleTable, memspace, file_space, H5P_DEFAULT, &part1);
    H5Sclose( file_space);

    file_space = H5Dget_space( final_1 );
    H5Sselect_hyperslab( file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    H5Dwrite(final_1, particleTable, memspace, file_space, H5P_DEFAULT, &part2);
    H5Sclose( file_space );

    file_space = H5Dget_space( final_2 );
    H5Sselect_hyperslab( file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    H5Dwrite(final_2, particleTable, memspace, file_space, H5P_DEFAULT, &part3);
    H5Sclose(file_space);

    file_space = H5Dget_space( properties );
    H5Sselect_hyperslab( file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    
    if (write_ranged){
        H5Dwrite(properties, rangedPropertiesTable, memspace, file_space, H5P_DEFAULT, &props);
    }else{
        H5Dwrite(properties, volumePropertiesTable, memspace, file_space, H5P_DEFAULT, &props);
    }
    

    // delete(&temp_data);

    // std::cout << "cleaning up" << std::endl;
    H5Sclose(file_space);
    H5Sclose(memspace);
    event_count++; 
}

void DataWriter::makeTables(){
    // Build the event properties table
    size_t dataSize = 0;
    dataSize += 8; // double - totalEnergy 
    dataSize += 8; // double - zenith 
    dataSize += 8; // double - azimuth 
    dataSize += 8; // double - finalStateX 
    dataSize += 8; // double - finalStateY 
    dataSize += 8; // int32_t - finalType1 // yeah, this should be 4 bytes. BUT hdf5 is /really/ dumb about data sizes 
    dataSize += 8; // int32_t - finalType2
    dataSize += 8; // int32_t - initialType
    dataSize += 8; // double - x
    dataSize += 8; // double - y
    dataSize += 8; // double - z
    dataSize += 8; // double - totalColumnDepth
    
    herr_t status; // hdf5 error type. 
    hid_t basicPropertiesTable = H5Tcreate(H5T_COMPOUND, dataSize);
    size_t offset = 0;
    hid_t real_long = H5Tcopy( H5T_NATIVE_LONG );
    status = H5Tset_size( real_long, 4);
    hid_t real_bool = H5Tcopy( H5T_NATIVE_HBOOL);
    status = H5Tset_size( real_bool, 1);
    hid_t real_little = H5Tcopy( H5T_NATIVE_UINT8 );
    status = H5Tset_size(real_little, 1 );

    status = H5Tinsert(basicPropertiesTable, "totalEnergy", HOFFSET(BasicEventProperties, totalEnergy), H5T_NATIVE_DOUBLE);
    status = H5Tinsert(basicPropertiesTable, "zenith", HOFFSET(BasicEventProperties, zenith), H5T_NATIVE_DOUBLE); 
    status = H5Tinsert(basicPropertiesTable, "azimuth", HOFFSET(BasicEventProperties, azimuth) , H5T_NATIVE_DOUBLE); 
    status = H5Tinsert(basicPropertiesTable, "finalStateX", HOFFSET(BasicEventProperties, finalStateX) , H5T_NATIVE_DOUBLE);
    status = H5Tinsert(basicPropertiesTable, "finalStateY", HOFFSET(BasicEventProperties, finalStateY) , H5T_NATIVE_DOUBLE); 
    status = H5Tinsert(basicPropertiesTable, "finalType1", HOFFSET(BasicEventProperties, finalType1) , real_long); 
    status = H5Tinsert(basicPropertiesTable, "finalType2", HOFFSET(BasicEventProperties, finalType2) , real_long);
    status = H5Tinsert(basicPropertiesTable, "initialType", HOFFSET(BasicEventProperties, initialType) , real_long); 
    status = H5Tinsert(basicPropertiesTable, "x", HOFFSET(BasicEventProperties, x) , H5T_NATIVE_DOUBLE); 
    status = H5Tinsert(basicPropertiesTable, "y", HOFFSET(BasicEventProperties, y) , H5T_NATIVE_DOUBLE); 
    status = H5Tinsert(basicPropertiesTable, "z", HOFFSET(BasicEventProperties, z) , H5T_NATIVE_DOUBLE); 
    status = H5Tinsert(basicPropertiesTable, "totalColumnDepth", HOFFSET(BasicEventProperties, totalColumnDepth) , H5T_NATIVE_DOUBLE); 

    // we want tables for volume and ranged, so let's copy that basic one and make the (slightly) different ones below
    rangedPropertiesTable = H5Tcopy( basicPropertiesTable );

    volumePropertiesTable = H5Tcopy( basicPropertiesTable );

    H5Tclose( basicPropertiesTable );

    hsize_t point3d_dim[1] = {3};
    hid_t   point3d     = H5Tarray_create( H5T_NATIVE_DOUBLE, 1, point3d_dim);
    hsize_t dir3d_dim[1] = {2};
    hid_t   direction   = H5Tarray_create( H5T_NATIVE_DOUBLE, 1, dir3d_dim);
    
    dataSize = 8 + 8 + 3*8 +2*8 +8 ; // native bool, int32, position, direction, energy = 64 bytes
    //  hdf5 has asinine datasizes. Everything is 8 bytes!!! 
    offset = 0;
    particleTable = H5Tcreate( H5T_COMPOUND, dataSize);
    status = H5Tinsert(particleTable, "initial",        HOFFSET(h5Particle, initial), real_bool);
    status = H5Tinsert(particleTable, "ParticleType",   HOFFSET(h5Particle, ptype),  real_long); 
    status = H5Tinsert(particleTable, "Position",       HOFFSET(h5Particle, pos) , point3d);     
    status = H5Tinsert(particleTable, "Direction",      HOFFSET(h5Particle, dir) , direction); 
    status = H5Tinsert(particleTable, "Energy",         HOFFSET(h5Particle, energy) , H5T_NATIVE_DOUBLE); 


}


} // end namespace LeptonInjector
