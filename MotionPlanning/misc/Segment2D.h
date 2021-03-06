/*
This file is part of LMPL.

    LMPL is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LMPL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the Lesser
    GNU General Public License for more details.

    You should have received a copy of the Lesser GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MATH3D_SEGMENT2D_H
#define MATH3D_SEGMENT2D_H

//#include "misc/GeometricBasics.h"
#include "AABB2D.h"
namespace MathGeometric {

/** @ingroup Math3D
 * @brief A 2D segment class
 *
 * Represented by the endpoints a and b.  Is parameterized by t in [0,1]
 * going from a->b as t approaches 1.
 */
struct Segment2D
{
	bool intersects(const AABB2D&) const;
	///given bounds [tmin,tmax] of the segment, returns the clipping min/max
	bool intersects(const AABB2D&, double& tmin, double& tmax) const;

  Point2D a,b;
};

} //namespace Math3D

#endif
