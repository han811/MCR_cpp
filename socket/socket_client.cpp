#include "ClientSocket.h"
#include "SocketException.h"
#include <iostream>
#include <string>

int main ( int argc, int argv[] )
{
  try
    {
      // Create the client socket
      ClientSocket client_socket ( "localhost", 30000 );

      // rest of code -
      // send request, retrieve reply, etc...

    }
  catch ( SocketException& e )
    {
      std::cout << "Exception was caught:" << e.description() << "\n";
    }

  return 0;
}

