const passport = require("passport");
const JWTstrategy = require("passport-jwt").Strategy;
const ExtractJWT = require("passport-jwt").ExtractJwt;
const env = require('./environment');
const User = require("../models/JavaScript1");


let opts = {
    jwtFromRequest : ExtractJWT.fromAuthHeaderAsBearerToken(),
    secretOrKey: env.jwt_secret,

}

passport.use(new JWTstrategy(opts, function (jwt_payload, done) {

    User.findById(jwt_payload._id, function (err, user) {

        if (err) {
            console.log("There was an error in extracting user from the DB");
            //return done(err);
            return;
        }
        if (user) {
            return done(null, user);
        }
        else {
            return done(null, false);
        }
    })

}));

module.exports = passport;
