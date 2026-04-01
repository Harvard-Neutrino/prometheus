#ifndef LI_RANDOM
#define LI_RANDOM

// Ben Smithers
// benjamin.smithers@mavs.uta.edu

// this implements a class to sample numbers just like in an i3 service

#include <random> // default_random_engine, uniform_real_distribution 

namespace LeptonInjector {

    class LI_random{
        public:
            LI_random();
            LI_random( unsigned int seed );

            // this naming convention is used to
            double Uniform( double from=0.0, double to=1.0);

            // in case this is set up without a seed! 
            void set_seed(unsigned int new_seed);

        private: 
            std::default_random_engine configuration;
            std::uniform_real_distribution<double> generator;
    };

} //end namespace LeptonInjector

#endif
