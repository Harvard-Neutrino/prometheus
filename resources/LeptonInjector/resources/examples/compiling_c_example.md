# The c++ example

Use GNU c++ compiler to compile this example! It needs to link to your compiled LeptonInjector shared object and to your photospline library. Simply run the command before from inside this folder. 

```
g++ -I../../public inject_muons.cpp -llepton_injector -lphotospline -I../../public/LeptonInjector -I../../public/earthmodel-service -I../../public/phys-services -o inject_muons.o
```

You can optionally append the "-g" flag to make better use of the g debugger. 
