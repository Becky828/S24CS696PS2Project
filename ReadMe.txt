SUMMARY
================================================================================

The ratings.csv file contains approximately 3,908,657 anonymous ratings of 
68,044 movies made by 6,724 MovieLens users who have logged in to the website 
over 12 times from 2019 to 2020 and rated over 20 movies since their registration.

USAGE LICENSE
================================================================================

Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.  The data set may be used for any research
purposes under the following conditions:

     * The user may not state or imply any endorsement from the
       University of Minnesota or the GroupLens Research Group.

     * The user must acknowledge the use of the data set in
       publications resulting from the use of the data set
       (see below for citation information).

     * The user may not redistribute the data without separate
       permission.

     * The user may not use this information for any commercial or
       revenue-bearing purposes without first obtaining permission
       from a faculty member of the GroupLens Research Project at the
       University of Minnesota.

If you have any further questions or comments, please contact GroupLens
<grouplens-info@cs.umn.edu>. 

CITATION
================================================================================

To acknowledge use of the dataset in publications, please cite the following
paper:

Ruixuan Sun, Ruoyan Kong, Qiao Jin, and Joseph A. Konstan. 2023. 
Less Can Be More: Exploring Population Rating Dispositions with Partitioned Models 
in Recommender Systems. In Adjunct Proceedings of the 31st ACM Conference on 
User Modeling, Adaptation and Personalization (UMAP ’23 Adjunct), June 26–29, 2023, 
Limassol, Cyprus. ACM, New York, NY, USA, 8 pages. 
https://doi.org/10.1145/3563359.3597390

ACKNOWLEDGEMENTS
================================================================================

Thanks to Ruoyan Kong and Daniel Kluver for cleaning up and generating the data
set.

FURTHER INFORMATION ABOUT THE GROUPLENS RESEARCH PROJECT
================================================================================

The GroupLens Research Project is a research group in the Department of 
Computer Science and Engineering at the University of Minnesota. Members of 
the GroupLens Research Project are involved in many research projects related 
to the fields of information filtering, collaborative filtering, and 
recommender systems. The project is lead by professors John Riedl and Joseph 
Konstan. The project began to explore automated collaborative filtering in 
1992, but is most well known for its world wide trial of an automated 
collaborative filtering system for Usenet news in 1996. Since then the project 
has expanded its scope to research overall information filtering solutions, 
integrating in content-based methods as well as improving current collaborative 
filtering technology.

Further information on the GroupLens Research project, including research 
publications, can be found at the following web site:
        
        http://www.grouplens.org/

GroupLens Research currently operates a movie recommender based on 
collaborative filtering:

        http://www.movielens.org/

FILE DESCRIPTION
================================================================================

All ratings are contained in the file "ratings.csv" and are in the
following format:

userId,movieId,rating,tstamp

- userId: the anonymized unique id for each active user, indexed from 1 to 6724.
- movieId: the id of the movie that the user (corresponding to userId) rated. 
Note this movie ID is the same as the one in other published movielens datasets.
- rating: the rating (from 0.5 to 5 stars) provided by the user.
- tstamp: when the user rated the movie.