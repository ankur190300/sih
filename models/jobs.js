
const mongoose = require('mongoose')

const jobSchema = new mongoose.Schema({
    description: {
        type: String,
        required: true
    },
    company: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
    },
    job_name: {
        type: String,
        required: true
    }, 
    salary:{
        type:Number, 
        required: true
    }, 
    
}, {
        timestamps: true
    });

const Job = mongoose.model('Job', jobSchema);

module.exports = Job;
