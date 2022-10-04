//You can use "cmake -DCGAL_DIR=$HOME/CGAL-5.4 -DCMAKE_BUILD_TYPE=Release ." in case if you don't have CGAL installed
//Have a look at "https://en.wikipedia.org/wiki/STL_(file_format)" for the syntax of STL file format (consider normal as 0,0,0)
//Use MeshLab software (https://www.meshlab.net/) to visualize the results incase you don't know how to render it

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Min_sphere_annulus_d_traits_3.h>
#include <CGAL/Min_sphere_d.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h> 
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef Delaunay::Point Point;
typedef CGAL::Min_sphere_annulus_d_traits_3<K>   Traits;
typedef CGAL::Min_sphere_d<Traits>               Min_sphere;


Delaunay T;
Point triangle_vertex[3];
//std::vector<Min_sphere> sphere_list;
std::vector<Delaunay::Triangle> new_triangle_list;
std::vector <double> radius_list;
//std::vector<std::array<double, 4> > triangle_vertex_list;
int vertex_it[4][3] = {1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4};

int main(int argc, char **argv)
{
    const char* fname =argv[1]; //Reading the filename from the arguments
    std::ifstream stream(fname); //Reading the file
    Point p;
    while(!stream.eof()) //While the file is completely read
    {
        stream>>p; //Save line by line (x,y,z coordinates) to a point variable
        T.insert(Point(p.x(),p.y(),p.z())); //Insert the point into incremental Delaunay construction
    }

    Delaunay::Finite_facets_iterator fit;
    Delaunay::Triangle test_triangle; 
    int total_triangles = 4*T.number_of_facets();
    //double triangle_list[total_triangles][3][3];
    //double *triangle_list = (double*)malloc(9*total_triangles*sizeof(double));
    // Min_sphere sphere_list[total_triangles];
    // double radius_list[total_triangles];

    double mean_radius;
    double stdev;
    double sq_sum;
    int removed_triangles;
    double alpha;
    double percentage;
    int triangle_counter = 0;
    int i;
    for(fit=T.finite_facets_begin();fit!=T.finite_facets_end();fit++){ //Do for each cell. T.finite_cells_begin() gives a starting tetrahedron, ++ moves to next tetrahedron, T.finite_cells_end() gives the final tetrahedron ** This loop is just to give an idea about how to access cells and vertices, it is not required in the program
    	//circum = CGAL::circumcenter(T.triangle(*fit));

        std::pair<Delaunay::Cell_handle, int> facet = *fit;
        for(i = 0; i!=4; ++i){
            Delaunay::Vertex_handle v1 = facet.first->vertex( (facet.second+vertex_it[i][0])%4);
            Delaunay::Vertex_handle v2 = facet.first->vertex( (facet.second+vertex_it[i][1])%4);
            Delaunay::Vertex_handle v3 = facet.first->vertex( (facet.second+vertex_it[i][2])%4);

            triangle_vertex[0] = v1 -> point();
            triangle_vertex[1] = v2 -> point();
            triangle_vertex[2] = v3 -> point();
            
            Min_sphere  ms(triangle_vertex, triangle_vertex+3);
            // sphere_list[triangle_counter] = ms;
            radius_list.push_back(std::sqrt(ms.squared_radius()));
            //radius_list[triangle_counter] = std::sqrt(ms.squared_radius());
        }
    }

    std::vector<double> diff(radius_list.size());
    mean_radius = std::accumulate(radius_list.begin(), radius_list.end(), 0.0) / radius_list.size();
    std::transform(radius_list.begin(), radius_list.end(), diff.begin(), [mean_radius](double x) { return x - mean_radius; });
    sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    stdev = std::sqrt(sq_sum / radius_list.size());

    std::cout<< "Mean radius = "<< mean_radius << "\n";
    std::cout<< "radius standard dev = "<< stdev <<  "\n";

    // // we should mantain the triangles which its sphere has radius < 1/alpha = mean_radius => alpha = 1/(mean_radius)
   

    alpha = 1/(mean_radius);
    cout << "Mean_radius = " << mean_radius;
    int remove_counter = 0;
    int new_triangles = 0;
    string filename = "result.STL";
    ofstream st;
    st.open(filename);
    st << "solid result\n";
    for(fit=T.finite_facets_begin();fit!=T.finite_facets_end();fit++){ //Do for each cell. T.finite_cells_begin() gives a starting tetrahedron, ++ moves to next tetrahedron, T.finite_cells_end() gives the final tetrahedron ** This loop is just to give an idea about how to access cells and vertices, it is not required in the program

        std::pair<Delaunay::Cell_handle, int> facet = *fit;
        for(i = 0; i<4; ++i){
            Delaunay::Vertex_handle v1 = facet.first->vertex( (facet.second+vertex_it[i][0])%4);
            Delaunay::Vertex_handle v2 = facet.first->vertex( (facet.second+vertex_it[i][1])%4);
            Delaunay::Vertex_handle v3 = facet.first->vertex( (facet.second+vertex_it[i][2])%4);
            triangle_vertex[0] = v1 -> point();
            triangle_vertex[1] = v2 -> point();
            triangle_vertex[2] = v3 -> point();

            if(radius_list[remove_counter] < 1/alpha){
                st << "facet normal 0 0 0\n";
                st << "outer loop\n";
                st << "vertex " << triangle_vertex[0] << "\n";
                st << "vertex " << triangle_vertex[1] << "\n";
                st << "vertex " << triangle_vertex[2] << "\n";
                st << "endloop\n";
                st << "endfacet\n";
                new_triangles++;
            }
            remove_counter++;
        }
        
    }
    st << "endsolid result\n";
    st.close();
    std::cout << filename <<" successfully saved." <<"\n";


    removed_triangles = total_triangles - new_triangles;
    percentage = ((1.0*removed_triangles/(1.0*total_triangles))*100.0);

    std::cout << "Total triangles before filtering = " << total_triangles << "\n";
    // std::cout << "1/alpha = " << 1/alpha << "\n";
    std::cout <<"Number of removed triangles = " << removed_triangles << " (" <<  percentage <<"%)"<< "\n";
    std::cout <<"Number of triangles after filtering = " << new_triangles << "\n";

    // std::cout << triangle_list[0] << "\n";

    // writeSTL(triangle_list, "input.STL");
    // writeSTL(new_triangle_list, "result.STL");

    //Printing the vertex coordinates of 4 points of the tetrahedron
    //TODO: Apply Delaunay filtering (take each Delaunay face [have a look at Delaunay facets as well - https://doc.cgal.org/latest/Triangulation_3/index.html], and check whether it satisfies the requirement)
    //Hint: You can use CGAL::circumcenter() to compute the circumcenters of a Triangle/Cell
    //TODO: Save the result into an STL file
    return 1;
}

