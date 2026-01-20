/*******************************************************************************
   2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



   Outputs: initial.dat - inital values of u(x,y)
            final.dat   - final values of u(x,y)

            The output files have three columns: x, y, u

            Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

   Notes: The time step is calculated using the CFL condition

 ********************************************************************************/

/*********************************************************************
                     Include header files
 **********************************************************************/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>

/*********************************************************************
                      Main function
 **********************************************************************/

float equation_one( float y ) {
    const float friction_vel = 0.2;
    const float z0           = 1.0;
    const float k            = 0.41; // von Karman constant
    // Only use Equation (1) if height is greater than z0 otherwise horizontal velocity is 0
    if( y > z0 )
    {
        return ( friction_vel / k ) * log( y / z0 );
    }
    else
    {
        return 0.0;
    }
}

int main(){

    /* Grid properties */
    const int   NX   = 1000; // Number of x points
    const int   NY   = 1000; // Number of y points
    const float xmin = 0.0; // Minimum x value
    const float xmax = 30.0; // Maximum x value /* Task 2: Change xmax to 30.0 */
    const float ymin = 0.0; // Minimum y value
    const float ymax = 30.0; // Maximum y value /* Task 2: Change ymax to 30.0 */

    /* Parameters for the Gaussian initial conditions */
    const float x0      = 3.0;           // Centre(x) /* Task 2: Change centre to 3.0 */
    const float y0      = 15.0;           // Centre(y) /* Task 2: Change centre to 15.0 */
    const float sigmax  = 1.0;          // Width(x)  /* Task 2: Change width to 1.0 */
    const float sigmay  = 5.0;          // Width(y)  /* Task 2: Change width to 5.0 */
    const float sigmax2 = sigmax * sigmax; // Width(x) squared
    const float sigmay2 = sigmay * sigmay; // Width(y) squared

    /* Boundary conditions */
    const float bval_left  = 0.0; // Left boudnary value
    const float bval_right = 0.0; // Right boundary value
    const float bval_lower = 0.0; // Lower boundary
    const float bval_upper = 0.0; // Upper bounary

    /* Time stepping parameters */
    const float CFL    = 0.9; // CFL number
    const int   nsteps = 800; // Number of time steps /* Task 2: Change nsteps to 800 */

    /* Velocity */
    const float velx = 1.0; // Velocity in x direction /* Task 2: Change velx to 1.0 */
    const float vely = 0.0; // Velocity in y direction /* Task 2: Change vely to 0.0 */
    /* Arrays to store variables. These have NX+2 elements
       to allow boundary values to be stored at both ends */
    float x[NX + 2];        // x-axis values
    float y[NX + 2];        // y-axis values
    float u[NX + 2][NY + 2];  // Array of u values
    float dudt[NX + 2][NY + 2]; // Rate of change of u

    float x2; // x squared (used to calculate iniital conditions)
    float y2; // y squared (used to calculate iniital conditions)

    /* Calculate distance between points */
    float dx = ( xmax - xmin ) / ( (float) NX );
    float dy = ( ymax - ymin ) / ( (float) NY );

    /* Calculate time step using the CFL condition */
    /* The fabs function gives the absolute value in case the velocity is -ve */
    /* Task 3: Changes max possible velocity so this needs updating */
    float dt = CFL / ( ( fabs( equation_one( ymax ) ) / dx ) + ( fabs( vely ) / dy ) );

    /*** Report information about the calculation ***/
    printf( "Grid spacing dx     = %g\n", dx );
    printf( "Grid spacing dy     = %g\n", dy );
    printf( "CFL number          = %g\n", CFL );
    printf( "Time step           = %g\n", dt );
    printf( "No. of time steps   = %d\n", nsteps );
    printf( "End time            = %g\n", dt * (float) nsteps );
    printf( "Distance advected x = %g\n", equation_one( ymax ) * dt * (float) nsteps );
    printf( "Distance advected y = %g\n", vely * dt * (float) nsteps );

    /* Task 1: Parallelise the loops using OpenMP when acceptable and reasonable to do so */
    /* Leave a comment explaining why a loop could not be parallelised */

    double start_time = 0.0;
    double end_time   = 0.0;
    start_time = omp_get_wtime();

    /*** Place x points in the middle of the cell ***/
    /* LOOP 1 */
    /* Minimal work, so not parallelised - thread start overhead too great */
    for( int i = 0; i < NX + 2; i++ )
    {
        x[i] = ( (float) i - 0.5 ) * dx;
    }

    /*** Place y points in the middle of the cell ***/
    /* LOOP 2 */
    /* Minimal work, so not parallelised - thread start overhead too great */
    for( int j = 0; j < NY + 2; j++ )
    {
        y[j] = ( (float) j - 0.5 ) * dy;
    }

    /*** Set up Gaussian initial conditions ***/
    /* LOOP 3 */
    #pragma omp parallel for collapse(2) private(x2,y2)
    for( int i = 0; i < NX + 2; i++ )
    {
        for( int j = 0; j < NY + 2; j++ )
        {
            /* Equation (6) */
            x2      = ( x[i] - x0 ) * ( x[i] - x0 );
            y2      = ( y[j] - y0 ) * ( y[j] - y0 );
            u[i][j] = exp( -1.0 * ( ( x2 / ( 2.0 * sigmax2 )) + ( y2 / ( 2.0 * sigmay2 )) ) );
        }
    }

    /*** Write array of initial u values out to file ***/
    FILE *initialfile;
    initialfile = fopen( "initial.dat", "w" );
    /* LOOP 4 */
    /* Cannot parallelise this loop because writing to a file is not thread-safe */
    for( int i = 0; i < NX + 2; i++ )
    {
        for( int j = 0; j < NY + 2; j++ )
        {
            fprintf( initialfile, "%g %g %g\n", x[i], y[j], u[i][j] );
        }
    }
    fclose( initialfile );

    /*** Update solution by looping over time steps ***/
    /* LOOP 5 */
    /* Cannot parallelise this loop because each iteration depends on the results of the previous iteration */
    for( int m = 0; m < nsteps; m++ )
    {
        #pragma omp parallel
        {
            /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
            /* LOOP 6 */
            #pragma omp for nowait
            for( int j = 0; j < NY + 2; j++ )
            {
                u[0][j]      = bval_left;
                u[NX + 1][j] = bval_right;
            }

            /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
            /* LOOP 7 */
            #pragma omp for nowait
            for( int i = 0; i < NX + 2; i++ )
            {
                u[i][0]      = bval_lower;
                u[i][NY + 1] = bval_upper;
            }
        }

        /*** Calculate rate of change of u using leftward difference ***/
        /* Loop over points in the domain but not boundary values */
        /* LOOP 8 */
        #pragma omp parallel for collapse(2)
        for( int i = 1; i < NX + 1; i++ )
        {
            for( int j = 1; j < NY + 1; j++ )
            {
                /* Task 3: replace velx with velx as calculated from Equation (1) */
                float velx_calc = equation_one( y[j] );

                /* Equation (2), (3) and (4) */
                dudt[i][j] = -velx_calc * ( u[i][j] - u[i - 1][j] ) / dx
                    - vely * ( u[i][j] - u[i][j - 1] ) / dy;
            }
        }

        /*** Update u from t to t+dt ***/
        /* Loop over points in the domain but not boundary values */
        /* LOOP 9 */
        #pragma omp parallel for collapse(2)
        for( int i = 1; i < NX + 1; i++ )
        {
            for( int j = 1; j < NY + 1; j++ )
            {
                /* Equation (5) */
                u[i][j] = u[i][j] + dudt[i][j] * dt;
            }
        }
    } // time loop

    end_time = omp_get_wtime();
    printf( "Elapsed time: %g seconds\n", end_time - start_time );

    /*** Write array of final u values out to file ***/
    FILE *finalfile;
    finalfile = fopen( "final.dat", "w" );
    /* LOOP 10 */
    /* Cannot parallelise this loop because writing to a file is not thread-safe */
    for( int i = 0; i < NX + 2; i++ )
    {
        for( int j = 0; j < NY + 2; j++ )
        {
            fprintf( finalfile, "%g %g %g\n", x[i], y[j], u[i][j] );
        }
    }

    fclose( finalfile );

    /* Task 4: Calculate the vertically averaged distribution of u(x,y) */
    /* The vertical average should cover the whole vertical domain, but not include the boundary values */
    /* Write output to a file for plotting with gnuplot */
    FILE *avgfile;
    avgfile = fopen( "vertical_average.dat", "w" );
    for( int i = 1; i < NX; i++ )
    {
        float sum = 0.0;
        for( int j = 1; j < NY; j++ )
        {
            sum += u[i][j];
        }
        float avg = sum / (float) ( NY - 2 ); // Exclude boundary values from average
        fprintf( avgfile, "%g %g\n", x[i], avg );
    }
    fclose( avgfile );

    return 0;
}

/* End of file ******************************************************/
