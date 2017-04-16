
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>

unsigned version = 0;


unsigned compute_seq_v0 (unsigned nb_iter);
unsigned compute_seq_v1 (unsigned nb_iter);
unsigned compute_seq_v2 (unsigned nb_iter);
unsigned compute_OMP_FOR_v0(unsigned nb_iter);
unsigned compute_OMP_FOR_v1(unsigned nb_iter);
unsigned compute_OMP_FOR_v2(unsigned nb_iter);

static int isAlive(int x, int y);
static int willBeAlive(int x, int y);
static void propagate();
void init_seq_v2();
static int max(int ,int);
static int min(int,int);

#define SIZE_TILE 32
#define NB_TILE DIM/SIZE_TILE

void_func_t first_touch [] = {
  NULL,
  NULL,
  NULL,
};

int_func_t compute [] = {
  compute_seq_v0,
  compute_seq_v1,
  compute_seq_v2,
  compute_OMP_FOR_v0,
  compute_OMP_FOR_v1,
  compute_OMP_FOR_v2,
};

char *version_name [] = {
  "Séquentielle",
  "Séquentielle Tuile",
  "Séquentielle Optimisé",
  "OpenMPFor de base",
  "OpenMPFor tuilé",
  "OpenMPFor optimisé"
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  0,
  0,
  0
};

///////////////////////////// Version séquentielle simple


unsigned compute_seq_v0 (unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it ++) {
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
	next_img (i, j) = willBeAlive(i,j);

    swap_images ();
  }
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}

unsigned compute_seq_v1(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it ++) {
    for(int x = 0; x<DIM ; x+=SIZE_TILE ){
      for(int y = 0; y<DIM ; y+=SIZE_TILE ){
        for (int i = x; i < x+SIZE_TILE; i++){
          for (int j = y; j < y+SIZE_TILE; j++){
            next_img (i, j) = willBeAlive(i,j);
          }
        }
      }
    }
    swap_images ();
  }
  return 0;
}

int **activeTile;
int **tmpActiveTile;

void init_seq_v2(){
  activeTile = malloc(sizeof(int*)*NB_TILE);
  tmpActiveTile = malloc(sizeof(int*)*NB_TILE);
  for(int i =0;i<NB_TILE;i++){
    activeTile[i]=malloc(sizeof(int)*NB_TILE);
    tmpActiveTile[i]=malloc(sizeof(int)*NB_TILE);
    for (int j = 0; j<NB_TILE;j++){
      activeTile[i][j] = 1;
    }
  }
}

static void propagate()
{
  for(int x =0;x<NB_TILE;x++){
    for (int y = 0; y<NB_TILE;y++){
      activeTile[x][y]=0;
      for(int i = max(x-1, 0); i <= min(x+1, NB_TILE-1); i++){
        for(int j = max(y-1, 0); j <= min(y+1, NB_TILE-1); j++){
          if (tmpActiveTile[i][j]==1) {
            activeTile[x][y]=1;
          }
        }
      }
    }
  }
}

unsigned compute_seq_v2(unsigned nb_iter)
{
  static int init = 1;
  if (init) {
    init_seq_v2();
    init =0;
  }

  for (unsigned it = 1; it <= nb_iter; it ++) {
    for(int x = 0; x<DIM ; x+=SIZE_TILE ){
      for(int y = 0; y<DIM ; y+=SIZE_TILE ){
        if (activeTile[x/SIZE_TILE][y/SIZE_TILE]) {
          tmpActiveTile[x/SIZE_TILE][y/SIZE_TILE]=0;
          for (int i = x; i < x+SIZE_TILE; i++){
            for (int j = y; j < y+SIZE_TILE; j++){
              next_img (i, j) = willBeAlive(i,j);
              if (cur_img(i,j)!= next_img(i,j)) {
                tmpActiveTile[x/SIZE_TILE][y/SIZE_TILE]=1;
              }
            }
          }
        }
      }
    }
    propagate();
    swap_images ();
  }
  return 0;
}



unsigned compute_OMP_FOR_v0(unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it ++) {
    #pragma omp parallel for
    for (int i = 0; i < DIM; i++){
      for (int j = 0; j < DIM; j++){
        next_img (i, j) = willBeAlive(i,j);
      }
    }

    swap_images ();
  }
  return 0;
}

unsigned compute_OMP_FOR_v1(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it ++) {
    #pragma omp parallel for
    for(int x = 0; x<DIM ; x+=SIZE_TILE ){
      for(int y = 0; y<DIM ; y+=SIZE_TILE ){
        for (int i = x; i < x+SIZE_TILE; i++){
          for (int j = y; j < y+SIZE_TILE; j++){
            next_img (i, j) = willBeAlive(i,j);
          }
        }
      }
    }
    swap_images ();
  }
  return 0;
}

unsigned compute_OMP_FOR_v2(unsigned nb_iter)
{
  static int init = 1;
  if (init) {
    init_seq_v2();
    init =0;
  }
  for (unsigned it = 1; it <= nb_iter; it ++) {
    #pragma omp parallel for
    for(int x = 0; x<DIM ; x+=SIZE_TILE ){
      for(int y = 0; y<DIM ; y+=SIZE_TILE ){
        if (activeTile[x/SIZE_TILE][y/SIZE_TILE]) {
          tmpActiveTile[x/SIZE_TILE][y/SIZE_TILE]=0;
          for (int i = x; i < x+SIZE_TILE; i++){
            for (int j = y; j < y+SIZE_TILE; j++){
              next_img (i, j) = willBeAlive(i,j);
              if (cur_img(i,j)!= next_img(i,j)) {
                tmpActiveTile[x/SIZE_TILE][y/SIZE_TILE]=1;
              }
            }
          }
        }
      }
    }
    propagate();
    swap_images ();
  }
  return 0;
}




static int max(int a, int b)
{
  return a>b ? a : b;
}

static int min(int a, int b)
{
  return a<b ? a : b;
}

static unsigned couleur = 0xFFFF00FF; // Yellow

 int willBeAlive(int x, int y)
{
  int nbAlive=0;
  for(int i = max(x-1, 0); i <= min(x+1, DIM-1); i++){
    for(int j = max(y-1, 0); j <= min(y+1, DIM-1); j++){
      if((i != x || j != y) && isAlive(i,j)){
        nbAlive++;
      }
    }
  }
  if(!isAlive(x,y) && nbAlive==3)
    return couleur;
  else if(isAlive(x, y) && (nbAlive==2 || nbAlive==3))
    return couleur;
  else
    return 0x00;
}

int isAlive(int x, int y){
  return cur_img(x,y) != 0;
}
