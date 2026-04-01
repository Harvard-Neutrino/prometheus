#ifndef LI_PARTICLE
#define LI_PARTICLE

// Used to define the Particle class
// Partiles have a type, energy, position, and direction

// !!! Important !!!
// At the moment, only leptons (charged + uncharged) and hadrons are fully supported 

#include <string>
#include <exception>
#include <utility> // std::pair

#include <LeptonInjector/Constants.h>
#include <cstdint>

// positions are in Cartesian, centered in the middle of IceCube

namespace LeptonInjector{ 
    

    // simple data structure for particles
    class Particle{
        public:

            // these match the PDG codes!
            // copied over from IceCube's dataclasses I3Particle definition
            enum ParticleType : int32_t{
                unknown = 0,
                Gamma = 22,
                EPlus = -11,
                EMinus = 11,
                MuPlus = -13,
                MuMinus = 13,
                Pi0 = 111,
                PiPlus = 211,
                PiMinus = -211,
                K0_Long = 130,
                KPlus = 321,
                KMinus = -321,
                Neutron = 2112,
                PPlus = 2212,
                PMinus = -2212,
                K0_Short = 310,
                Eta = 221,
                Lambda = 3122,
                SigmaPlus = 3222,
                Sigma0 = 3212,
                SigmaMinus = 3112,
                Xi0 = 3322,
                XiMinus = 3312,
                OmegaMinus = 3334,
                NeutronBar = -2112,
                LambdaBar = -3122,
                SigmaMinusBar = -3222,
                Sigma0Bar = -3212,
                SigmaPlusBar = -3112,
                Xi0Bar = -3322,
                XiPlusBar = -3312,
                OmegaPlusBar = -3334,
                DPlus = 411,
                DMinus = -411,
                D0 = 421,
                D0Bar = -421,
                DsPlus = 431,
                DsMinusBar = -431,
                LambdacPlus = 4122,
                WPlus = 24,
                WMinus = -24,
                Z0 = 23,
                NuE = 12,
                NuEBar = -12,
                NuMu = 14,
                NuMuBar = -14,
                TauPlus = -15,
                TauMinus = 15,
                NuTau = 16,
                NuTauBar = -16,
                
                /* Nuclei */
                H2Nucleus = 1000010020,
                He3Nucleus = 1000020030,
                He4Nucleus = 1000020040,
                Li6Nucleus = 1000030060,
                Li7Nucleus = 1000030070,
                Be9Nucleus = 1000040090,
                B10Nucleus = 1000050100,
                B11Nucleus = 1000050110,
                C12Nucleus = 1000060120,
                C13Nucleus = 1000060130,
                N14Nucleus = 1000070140,
                N15Nucleus = 1000070150,
                O16Nucleus = 1000080160,
                O17Nucleus = 1000080170,
                O18Nucleus = 1000080180,
                F19Nucleus = 1000090190,
                Ne20Nucleus = 1000100200,
                Ne21Nucleus = 1000100210,
                Ne22Nucleus = 1000100220,
                Na23Nucleus = 1000110230,
                Mg24Nucleus = 1000120240,
                Mg25Nucleus = 1000120250,
                Mg26Nucleus = 1000120260,
                Al26Nucleus = 1000130260,
                Al27Nucleus = 1000130270,
                Si28Nucleus = 1000140280,
                Si29Nucleus = 1000140290,
                Si30Nucleus = 1000140300,
                Si31Nucleus = 1000140310,
                Si32Nucleus = 1000140320,
                P31Nucleus = 1000150310,
                P32Nucleus = 1000150320,
                P33Nucleus = 1000150330,
                S32Nucleus = 1000160320,
                S33Nucleus = 1000160330,
                S34Nucleus = 1000160340,
                S35Nucleus = 1000160350,
                S36Nucleus = 1000160360,
                Cl35Nucleus = 1000170350,
                Cl36Nucleus = 1000170360,
                Cl37Nucleus = 1000170370,
                Ar36Nucleus = 1000180360,
                Ar37Nucleus = 1000180370,
                Ar38Nucleus = 1000180380,
                Ar39Nucleus = 1000180390,
                Ar40Nucleus = 1000180400,
                Ar41Nucleus = 1000180410,
                Ar42Nucleus = 1000180420,
                K39Nucleus = 1000190390,
                K40Nucleus = 1000190400,
                K41Nucleus = 1000190410,
                Ca40Nucleus = 1000200400,
                Ca41Nucleus = 1000200410,
                Ca42Nucleus = 1000200420,
                Ca43Nucleus = 1000200430,
                Ca44Nucleus = 1000200440,
                Ca45Nucleus = 1000200450,
                Ca46Nucleus = 1000200460,
                Ca47Nucleus = 1000200470,
                Ca48Nucleus = 1000200480,
                Sc44Nucleus = 1000210440,
                Sc45Nucleus = 1000210450,
                Sc46Nucleus = 1000210460,
                Sc47Nucleus = 1000210470,
                Sc48Nucleus = 1000210480,
                Ti44Nucleus = 1000220440,
                Ti45Nucleus = 1000220450,
                Ti46Nucleus = 1000220460,
                Ti47Nucleus = 1000220470,
                Ti48Nucleus = 1000220480,
                Ti49Nucleus = 1000220490,
                Ti50Nucleus = 1000220500,
                V48Nucleus = 1000230480,
                V49Nucleus = 1000230490,
                V50Nucleus = 1000230500,
                V51Nucleus = 1000230510,
                Cr50Nucleus = 1000240500,
                Cr51Nucleus = 1000240510,
                Cr52Nucleus = 1000240520,
                Cr53Nucleus = 1000240530,
                Cr54Nucleus = 1000240540,
                Mn52Nucleus = 1000250520,
                Mn53Nucleus = 1000250530,
                Mn54Nucleus = 1000250540,
                Mn55Nucleus = 1000250550,
                Fe54Nucleus = 1000260540,
                Fe55Nucleus = 1000260550,
                Fe56Nucleus = 1000260560,
                Fe57Nucleus = 1000260570,
                Fe58Nucleus = 1000260580,

                /* Exotics */
                Qball = 10000000,

                /* The following are fake particles used in Icetray and have no official codes */
                /* The section abs(code) > 2000000000 is reserved for this kind of use */
                CherenkovPhoton = 2000009900,
                Nu = -2000000004,
                Monopole = -2000000041,
                Brems = -2000001001,
                DeltaE = -2000001002,
                PairProd = -2000001003,
                NuclInt = -2000001004,
                MuPair = -2000001005,
                Hadrons = -2000001006,
                ContinuousEnergyLoss = -2000001111,
                FiberLaser = -2000002100,
                N2Laser = -2000002101,
                YAGLaser = -2000002201,
                STauPlus = -2000009131,
                STauMinus = -2000009132,
                SMPPlus = -2000009500,
                SMPMinus = -2000009501,

            };

            // There are two kinds of event topologies in IceCube
            //      cascades 
            //      tracks
            enum class ParticleShape{ MCTrack, Cascade, unknown };

        public:

            Particle();

            Particle(ParticleType type);

            
            // what kind of particle is this (see below)
            ParticleType type; 
            // what is this event's topology? (see below)
        
            double energy; // GeV 
            std::pair<double, double> direction; //( zenith, azimuth ) in degrees
            double position[3]; // (x,y,z) in meters
        
            double GetMass(); //GeV/c^2
            bool HasMass(); // .... 
            std::string GetTypeString();
        private:
            int32_t pdgEncoding_; 

    };

    // prototype some of the particle helper functions

    bool isLepton(Particle::ParticleType p);
    bool isCharged(Particle::ParticleType p);
    std::string particleName( Particle::ParticleType p);
    double particleMass( Particle::ParticleType type);
    double kineticEnergy( Particle::ParticleType type, double totalEnergy);
    double particleSpeed( Particle::ParticleType type, double kineticEnergy);
    Particle::ParticleShape decideShape(Particle::ParticleType t);
    Particle::ParticleType deduceInitialType( Particle::ParticleType pType1, Particle::ParticleType pType2);
    uint8_t getInteraction( Particle::ParticleType final_1 , Particle::ParticleType final_2);

}// end namespace LI_Particle

#define PARTICLE_H_Particle_ParticleType                                      \
    (unknown)(Gamma)(EPlus)(EMinus)(MuPlus)(MuMinus)(Pi0) \
    (PiPlus)(PiMinus)(K0_Long)(KPlus)(KMinus)(Neutron)(PPlus)(PMinus)(K0_Short)   \
    (Eta)(Lambda)(SigmaPlus)(Sigma0)(SigmaMinus)(Xi0)(XiMinus)(OmegaMinus)        \
    (NeutronBar)(LambdaBar)(SigmaMinusBar)(Sigma0Bar)(SigmaPlusBar)(Xi0Bar)       \
    (XiPlusBar)(OmegaPlusBar)(DPlus)(DMinus)(D0)(D0Bar)(DsPlus)(DsMinusBar)       \
    (LambdacPlus)(WPlus)(WMinus)(Z0)(NuE)(NuEBar)                                 \
    (NuMu)(NuMuBar)(TauPlus)(TauMinus)(NuTau)(NuTauBar)(H2Nucleus)                \
    (He3Nucleus)(He4Nucleus)(Li6Nucleus)(Li7Nucleus)(Be9Nucleus)(B10Nucleus)      \
    (B11Nucleus)(C12Nucleus)(C13Nucleus)(N14Nucleus)(N15Nucleus)(O16Nucleus)      \
    (O17Nucleus)(O18Nucleus)(F19Nucleus)(Ne20Nucleus)(Ne21Nucleus)(Ne22Nucleus)   \
    (Na23Nucleus)(Mg24Nucleus)(Mg25Nucleus)(Mg26Nucleus)(Al26Nucleus)(Al27Nucleus)\
    (Si28Nucleus)(Si29Nucleus)(Si30Nucleus)(Si31Nucleus)(Si32Nucleus)(P31Nucleus) \
    (P32Nucleus)(P33Nucleus)(S32Nucleus)(S33Nucleus)(S34Nucleus)(S35Nucleus)      \
    (S36Nucleus)(Cl35Nucleus)(Cl36Nucleus)(Cl37Nucleus)(Ar36Nucleus)(Ar37Nucleus) \
    (Ar38Nucleus)(Ar39Nucleus)(Ar40Nucleus)(Ar41Nucleus)(Ar42Nucleus)(K39Nucleus) \
    (K40Nucleus)(K41Nucleus)(Ca40Nucleus)(Ca41Nucleus)(Ca42Nucleus)(Ca43Nucleus)  \
    (Ca44Nucleus)(Ca45Nucleus)(Ca46Nucleus)(Ca47Nucleus)(Ca48Nucleus)(Sc44Nucleus)\
    (Sc45Nucleus)(Sc46Nucleus)(Sc47Nucleus)(Sc48Nucleus)(Ti44Nucleus)(Ti45Nucleus)\
    (Ti46Nucleus)(Ti47Nucleus)(Ti48Nucleus)(Ti49Nucleus)(Ti50Nucleus)(V48Nucleus) \
    (V49Nucleus)(V50Nucleus)(V51Nucleus)(Cr50Nucleus)(Cr51Nucleus)(Cr52Nucleus)   \
    (Cr53Nucleus)(Cr54Nucleus)(Mn52Nucleus)(Mn53Nucleus)(Mn54Nucleus)(Mn55Nucleus)\
    (Fe54Nucleus)(Fe55Nucleus)(Fe56Nucleus)(Fe57Nucleus)(Fe58Nucleus)(Qball)      \
    (CherenkovPhoton)(Nu)(Monopole)(Brems)(DeltaE)(PairProd)(NuclInt)(MuPair)     \
    (Hadrons)(ContinuousEnergyLoss)(FiberLaser)(N2Laser)(YAGLaser)                \


#endif
