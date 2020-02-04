const User = require('../../../models/JavaScript1');
const jwt = require("jsonwebtoken");
const env = require("../../../config/environment");

//function to sign the user in 
module.exports.createSession = async function (req, res) {

    try {
        let user = await User.findOne({ email: req.body.email });

        if (!user ) {
            return res.json(200, {
                message: "Invalid username"
            })
        }
        else if (user.password != req.body.password) {
            
            return res.json(200, {
                message: "Invalid password"
            })
        }
        else {
            return res.json(400, {
                message: "Sign in successful , here is your token , please keep it safe ", 
                data: {
                    token: jwt.sign(user.toJSON(), env.jwt_secret, { expiresIn: '100000' })
                }
            })
        }

    }
    catch (err) {
       // console.log("*****",err);
        return res.json(500, {
            message: "Internal server error"
        })
    }

    

    
};