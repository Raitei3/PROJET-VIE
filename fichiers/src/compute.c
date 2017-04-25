
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"
#include "constants.h"

#include <stdbool.h>

unsigned version = 0;


unsigned compute_seq_v0 (unsigned nb_iter);
unsigned compute_seq_v1 (unsigned nb_iter);
unsigned compute_seq_v2 (unsigned nb_iter);
unsigned compute_OMP_FOR_v0(unsigned nb_iter);
unsigned compute_OMP_FOR_v1(unsigned nb_iter);
unsigned compute_OMP_FOR_v2(unsigned nb_iter);
unsigned compute_OMP_TASK_v0(unsigned nb_iter);
unsigned compute_OMP_TASK_v1(unsigned nb_iter);
unsigned compute_opencl(unsigned nb_iter);

static int isAlive(int x, int y);
static int willBeAlive(int x, int y);
static void propagate_seq();
void init_seq_v2();
static int max(int ,int);
static int min(int,int);
void task_v0(int x, int y);
void task_v1(int x, int y);
static void propagate_omp_for();
static void propagate_omp_task();
static void propagate_task(int x , int y);

int **activeTile;
int **tmpActiveTile;

int size_tile = DEFAULT_TILE_SIZE;
#define NB_TILE DIM/size_tile

void_func_t first_touch [] = {
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL
};

int_func_t compute [] = {
  compute_seq_v0,
  compute_seq_v1,
  compute_seq_v2,
  compute_OMP_FOR_v0,
  compute_OMP_FOR_v1,
  compute_OMP_FOR_v2,
  compute_OMP_TASK_v0,
  compute_OMP_TASK_v1,
  compute_opencl
};

char *version_name [] = {
  "Séquentielle",
  "Séquentielle Tuile",
  "Séquentielle Optimisé",
  "OpenMPFor de base",
  "OpenMPFor tuilé",
  "OpenMPFor optimisé",
  "OpenMPTask tuilé",
  "OpenMPTask optimisé",
  "opencl"
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1
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
    for(int x = 0; x<DIM ; x+=size_tile ){
      for(int y = 0; y<DIM ; y+=size_tile ){
        for (int i = x; i < x+size_tile; i++){
          for (int j = y; j < y+size_tile; j++){
            next_img (i, j) = willBeAlive(i,j);
          }
        }
      }
    }
    swap_images ();
  }
  return 0;
}



unsigned compute_seq_v2(unsigned nb_iter)
{
  static int init = 1;
  if (init) {
    init_seq_v2();
    init =0;
  }

  for (unsigned it = 1; it <= nb_iter; it ++) {
    for(int x = 0; x<DIM ; x+=size_tile ){
      for(int y = 0; y<DIM ; y+=size_tile ){
        if (activeTile[x/size_tile][y/size_tile]) {
          tmpActiveTile[x/size_tile][y/size_tile]=0;
          for (int i = x; i < x+size_tile; i++){
            for (int j = y; j < y+size_tile; j++){
              next_img (i, j) = willBeAlive(i,j);
              if (cur_img(i,j)!= next_img(i,j)) {
                tmpActiveTile[x/size_tile][y/size_tile]=1;
              }
            }
          }
        }
      }
    }
    propagate_seq();
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
    for(int x = 0; x<DIM ; x+=size_tile ){
      for(int y = 0; y<DIM ; y+=size_tile ){
        for (int i = x; i < x+size_tile; i++){
          for (int j = y; j < y+size_tile; j++){
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
    for(int x = 0; x<DIM ; x+=size_tile ){
      for(int y = 0; y<DIM ; y+=size_tile ){
        if (activeTile[x/size_tile][y/size_tile]) {
          tmpActiveTile[x/size_tile][y/size_tile]=0;
          for (int i = x; i < x+size_tile; i++){
            for (int j = y; j < y+size_tile; j++){
              next_img (i, j) = willBeAlive(i,j);
              if (cur_img(i,j)!= next_img(i,j)) {
                tmpActiveTile[x/size_tile][y/size_tile]=1;
              }
            }
          }
        }
      }
    }
    propagate_omp_for();
    swap_images ();
  }
  return 0;
}


unsigned compute_OMP_TASK_v0(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it ++) {
    #pragma omp parallel
    #pragma omp single
    for(int x = 0; x<DIM ; x+=size_tile ){
      for(int y = 0; y<DIM ; y+=size_tile ){
        #pragma omp task
        task_v0(x,y);
      }
    }
    swap_images ();
  }
  return 0;
}

unsigned compute_OMP_TASK_v1(unsigned nb_iter)
{
  static int init = 1;
  if (init) {
    init_seq_v2();
    init =0;
  }

  for (unsigned it = 1; it <= nb_iter; it ++) {
    #pragma omp parallel
    #pragma omp single
    for(int x = 0; x<DIM ; x+=size_tile ){
      for(int y = 0; y<DIM ; y+=size_tile ){
        #pragma omp task
        task_v1(x,y);
        }
      }
    propagate_seq();
    swap_images ();
    }

  return 0;
}

unsigned compute_opencl(unsigned nb_iter)
{
  return ocl_compute(nb_iter);
}

void task_v0(int x, int y)
{
  for (int i = x; i < x+size_tile; i++){
    for (int j = y; j < y+size_tile; j++){
      next_img (i, j) = willBeAlive(i,j);
    }
  }
}

void task_v1(int x, int y)
{
  if (activeTile[x/size_tile][y/size_tile]) {
    tmpActiveTile[x/size_tile][y/size_tile]=0;
    for (int i = x; i < x+size_tile; i++){
      for (int j = y; j < y+size_tile; j++){
        next_img (i, j) = willBeAlive(i,j);
        if (cur_img(i,j)!= next_img(i,j)) {
          tmpActiveTile[x/size_tile][y/size_tile]=1;
        }
      }
    }
  }
}



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



static void propagate_seq()
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

static void propagate_omp_for()
{
  #pragma omp for
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

static void propagate_omp_task()
{
  #pragma omp parallel
  #pragma omp single
  for(int x =0;x<NB_TILE;x++){
    for (int y = 0; y<NB_TILE;y++){
      #pragma omp task
      propagate_task(x,y);
    }
  }
}

static void propagate_task(int x , int y)
{
  activeTile[x][y]=0;
  for(int i = max(x-1, 0); i <= min(x+1, NB_TILE-1); i++){
    for(int j = max(y-1, 0); j <= min(y+1, NB_TILE-1); j++){
      if (tmpActiveTile[i][j]==1) {
        activeTile[x][y]=1;
      }
    }
  }
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
