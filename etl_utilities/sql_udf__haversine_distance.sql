/*
	Simple UDF for computing great circle distance between two lat lon coordinates
*/

CREATE or REPLACE FUNCTION haversine_distance(
                                    origin_lat float,
                                    origin_lon float,
                                    destination_lat float,
                                    destination_lon float
                                  )
RETURNS FLOAT as
$$
  2 * 3961 * asin(sqrt( SQUARE( (sin(radians((destination_lat - origin_lat) / 2))) )  + cos(radians(origin_lat)) * cos(radians(destination_lat)) * SQUARE( (sin(radians((destination_lon - origin_lon) / 2))) ) ))
$$
;


