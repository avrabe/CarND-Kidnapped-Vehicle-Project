/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cassert>

#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;

double_t w_max;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 3;

    std::normal_distribution<double_t> N_x(x, std[0]);
    std::normal_distribution<double_t> N_y(y, std[1]);
    std::normal_distribution<double_t> N_theta(theta, std[2]);
    for (int i = 0; i < num_particles; i++) {

        Particle particle;
        particle.id = i;
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight = 1.0;

        particles.push_back(particle);
        weights.push_back(1.0);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    for (auto &&particle: particles) {
        double_t new_x;
        double_t new_y;
        double_t new_theta;
        if (yaw_rate != 0) {
            new_x = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            new_y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            new_theta = particle.theta + yaw_rate * delta_t;
        } else {
            new_x = particle.x + velocity * cos(particle.theta) * delta_t;
            new_y = particle.y + velocity * sin(particle.theta) * delta_t;
            // particle.theta = particle.theta
        }

        std::normal_distribution<double_t> N_x(new_x, std_pos[0]);
        std::normal_distribution<double_t> N_y(new_y, std_pos[1]);
        std::normal_distribution<double_t> N_theta(new_theta, std_pos[2]);
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html


    for (auto &&particle: particles) {
        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;
        __long_double_t weight = 1.0;
        w_max = std::numeric_limits<double>::min();

        double_t cos_p_t = cos(particle.theta);
        double_t sin_p_t = sin(particle.theta);
        for (auto &&observation: observations) {
            // translate the observation into MAP coordinates

            double_t tobs_x = particle.x + (observation.x * cos_p_t - observation.y * sin_p_t);
            double_t tobs_y = particle.y + (observation.x * sin_p_t + observation.y * cos_p_t);

            // nearest neighbor search
            __long_double_t min_dist = std::numeric_limits<__long_double_t>::max();
            __long_double_t distance = sensor_range;
            Map::single_landmark_s closest_landmark;
            int association = -1;
            for (auto &landmark : map_landmarks.landmark_list) {
                //distance = sqrt(pow(landmark.x_f - tobs_x, 2.0) + pow(landmark.y_f - tobs_y, 2.0));

                double d = dist(landmark.x_f, landmark.y_f, tobs_x, tobs_y);
                if (d < min_dist && d < sensor_range) {
                    min_dist = d;
                    association = landmark.id_i;
                    closest_landmark = landmark;
                }

            }
            if (association >= 0) {
                assert(closest_landmark.id_i == association);
                double min_x = closest_landmark.x_f;
                double min_y = closest_landmark.y_f;

                __long_double_t f = 2 * M_PI * std_landmark[0] * std_landmark[1];
                __long_double_t xx = 0.5 * pow(min_x - tobs_x, 2.0) / pow(std_landmark[0], 2.0);
                __long_double_t yy = 0.5 * pow(min_y - tobs_y, 2.0) / pow(std_landmark[1], 2.0);

                f = exp(-1 * (xx + yy)) / f;
/*
                cout << particle.id << " " << f;
                cout << " mvg(  " << min_x << ", "  << min_y ;
                cout << " , " << tobs_x << ", " << tobs_y << ")" << endl;
*/
                if (f > 0.1e-1000 && f < std::numeric_limits<__long_double_t>::max()) {
                    weight *= f;
                    //assert(weight > 0.0 && weight < 100.0);
                }
                associations.push_back(association);
                sense_x.push_back(tobs_x);
                sense_y.push_back(tobs_y);
                weights.push_back(f);
            }
        }
        particle.weight = weight;
        w_max = (max)(w_max, particle.weight);

        // Don't use SetAssociations as I modify the particle directly.
        particle.associations.clear();
        particle.sense_x.clear();
        particle.sense_y.clear();

        particle.associations = associations;
        particle.sense_x = sense_x;
        particle.sense_y = sense_y;

    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> new_particles;
/*
    for (auto &&particle: particles) {
        cout << particle.weight << " ";
    }
    cout << endl;
*/
    int index = 0;
    double_t beta = 0.0;
    std::uniform_real_distribution<double> distribution(0.0, 2 * w_max);
    for (int i = 0; i < num_particles; i++) {
        beta += distribution(gen);
        while (particles[index].weight < beta) {
            beta = beta - (particles[index].weight > 0 ? particles[index].weight : 0.0000001);
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;
    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
    }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
