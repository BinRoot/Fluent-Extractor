#include "FluentCalculator.h"

using namespace std;
using namespace pcl;

int main(int argc, char** argv) {
    PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);

    if (io::loadPCDFile<PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    cout << "thickness: " << FluentCalculator::thickness(cloud) << endl;
    cout << "y_symmetry: " << FluentCalculator::y_symmetry(cloud) << endl;
    cout << "x_symmetry: " << FluentCalculator::x_symmetry(cloud) << endl;
    cout << "height: " << FluentCalculator::height(cloud) << endl;
    cout << "width: " << FluentCalculator::width(cloud) << endl;
    vector<PointXYZ> outerBox = FluentCalculator::outerBoundingBox(cloud);
	cout << "outer bounding box: " << outerBox[0] << " " << outerBox[1] << endl;

    return 0;
}
