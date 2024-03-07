#include "gtest/gtest.h"
#include <iostream>

class MyEnvironment : public ::testing::Environment
{
public:
  void SetUp() override
  {
  }
};

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MyEnvironment);
  return RUN_ALL_TESTS();
}