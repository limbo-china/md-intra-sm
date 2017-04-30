#include "potential.h"
#include "cell.h"
#include "atom.h"
#include "system.h"
#include "timer.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// 初始化势函数结构体
void initPotInfo(Potential** pot){


	*pot = (Potential*)malloc(sizeof(Potential));
	Potential* potential= *pot;
	
	strcpy(potential->potentialType,"Morse");
	 // potential->De = 0.3429;	                  
	 // 	potential->re = 2.866;
		// potential->Beta = 1.3588;
	  	potential->cutoff = 5.7875;
	potential->sigma = 2.315;	                  // Angstrom
   potential->epsilon = 0.167;

		//potential->computeforce = computeForce;
		//potential->free = potentialFree;
}

// 释放结构体空间
void potentialFree(Potential* potential){
	if(potential)
		free(potential);
}

// 根据势函数，求原子间的相互作用力, 选取morse势函数
void  computeForce(struct SystemStr* sys){

	Potential* potential = sys->potential;
		//  double De = potential->De;
		//  double Beta = potential->Beta;
		// double re = potential->re;
		//  double cutoff = potential->cutoff;
	double sigma = potential->sigma;
   double epsilon = potential->epsilon;
   double rCut = potential->cutoff;
   double rCut2 = rCut*rCut;

    double s6 = sigma*sigma*sigma*sigma*sigma*sigma;

   double rCut6 = s6 / (rCut2*rCut2*rCut2);

		Cell* cells = sys->cells;
	Atom* atoms = sys->atoms;

   	// 力置0
   	for(int i=0; i<cells->totalCellNum*MAXPERCELL; i++)
   		for(int j=0;j<3;j++)
      		atoms->force[i][j] = 0.0;
   
   //real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;
   // real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
   // real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);
      	//printf("myatomnum: %d\n",sys->atoms->myNum);
      	//beginTimer(force);
      	//int calls1=0,calls2=0,calls3=0,calls4=0,calls5=0,calls6=0,calls7=0;
   	for (int cell1 = 0; cell1<cells->myCellNum; cell1++)
   	{
   		//calls1++;
      	int atomnum1 = cells->atomNum[cell1];
      	if ( atomnum1 == 0 ) 
      		continue;

      	//calls2++;
      	int3 cell1xyz,cell2xyz;
      	
      	getXYZByCell(cells,cell1xyz,cell1);

   		for(cell2xyz[0]=cell1xyz[0]-1;cell2xyz[0]<=cell1xyz[0]+1;cell2xyz[0]++)
   			for(cell2xyz[1]=cell1xyz[1]-1;cell2xyz[1]<=cell1xyz[1]+1;cell2xyz[1]++)
   				for(cell2xyz[2]=cell1xyz[2]-1;cell2xyz[2]<=cell1xyz[2]+1;cell2xyz[2]++)
   				{
   					//calls3++;
   					
   					int cell2 = findCellByXYZ(cells,cell2xyz);
	
   					int atomnum2 = cells->atomNum[cell2];
   					if ( atomnum2 == 0 ) 
      					continue;

      				//calls4++;
					//beginTimer(test);
      				for (int n1=cell1*MAXPERCELL,count1=0; count1<atomnum1; count1++,n1++)
         			{
         					int id1 = atoms->id[n1];
         				
         				for (int n2=cell2*MAXPERCELL,count2=0; count2<atomnum2; count2++,n2++)
            			{
            				
            				//calls5++;
            				int id2 = atoms->id[n2];

           					if (cell2 < cells->myCellNum && id2 <= id1 ){ // <=  or < ???
                  				continue; // 防止重复计算
                  			} 

                  			double3 r_vector;
           					double r_scalar = 0.0;
                  			//calls6++;
                  			for (int i=0; i<3; i++)
               				{ 
                  				r_vector[i] = atoms->pos[n1][i]-atoms->pos[n2][i];
                  				r_scalar += r_vector[i]*r_vector[i];
               				}

               				if ( r_scalar > rCut2/*cutoff*cutoff*/) {
               					continue;
               				}
               				//calls7++;
               				r_scalar = 1.0/r_scalar;
               				double r6 = s6 * (r_scalar*r_scalar*r_scalar);

               				double fr = - 4.0*epsilon*r6*r_scalar*(12.0*r6 - 6.0);
              				 for (int m=0; m<3; m++)
               				{
                  				atoms->force[n1][m] -= r_vector[m]*fr;
                  				atoms->force[n2][m] += r_vector[m]*fr;
               				}
               				//beginTimer(force);
               				 // r_scalar = sqrt(r_scalar);

               				 // double force_scalar = 0.0;

               				 // double t = 1.0/(exp(Beta*(r_scalar-re)));
               				 // force_scalar = 2*Beta*De*(t-t*t);
 
               				 // for (int i=0; i<3; i++)
               				 // {
                  	 // 			atoms->force[n1][i] -= (r_vector[i]/r_scalar)*force_scalar;
                  	 // 			atoms->force[n2][i] += (r_vector[i]/r_scalar)*force_scalar;
               				 // } 
               				//endTimer(force); 
   						}  
   						    
            		}
            		
            		//endTimer(test);
         		}
         		//printf("calls:%d\n",calls );
    }
    //endTimer(force);
	//printf("calls1: %d calls2: %d calls3: %d calls4: %d calls5: %d calls6: %d calls7: %d\n",calls1,calls2,calls3,calls4,calls5,calls6,calls7);
}