#include "Subset.h"

using namespace std;

int main(void)
{

    cout << "hi" << '\n';
    Subset s = Subset(3);
    s.insert_end(2);
    s.insert(1);

    cout << s << '\n';

    return 0;
}