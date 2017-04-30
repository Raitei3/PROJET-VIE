
export OMP_NUM_THREADS

if (($# >= 1));then
    REMOTE_NAME=$1
    echo launched on $REMOTE_NAME
else
    echo launched on localhost
fi
ITE=$(seq 10) # nombre de mesures
  
THREADS=$(seq 2 2 24) # nombre de threads

PARAM="-n" # parametres commun à toutes les executions

ITERATIONS="100"

SIZE="256
1024
4096"

VERSION="5"

SCHEDULE="static
dynamic"

TILE_SIZE_OMP="32
64
128"

TILE_SIZE_OPENCL="16
32"

execute (){
    for v in $VERSION; do
        for s in $SIZE; do
            for ite in $ITERATIONS; do
                for t in $TILE_SIZE_OMP; do
                    for OMP_SCHEDULE in $SCHEDULE; do
                    EXE="./prog $* $PARAM -s $s -i $ite -t $t -v $v"
                    OUTPUT="$(echo $EXE $OMP_SCHEDULE| tr -d ' ')"
                    echo $EXE
                    echo $OMP_SCHEDULE
                    echo $OUTPUT
                    for nb in $ITE; do
                        for OMP_NUM_THREADS in $THREADS; do
                            echo -n "$OMP_NUM_THREADS " >> $OUTPUT ;
                            echo "$OMP_NUM_THREADS thread, lancer n°$nb [$(date)]"
                            if [ -z $REMOTE_NAME ] ; then
                                $EXE 2>> $OUTPUT >/dev/null;
                            else
                                ssh -X -oProxyCommand="ssh fac nc %h %p" paubeziau@$REMOTE_NAME $EXE 2>> $OUTPUT >/dev/null;
                            fi
                        done;
                    done;
                    done;
                done;
            done
        done
    done
}


#execute -a
execute




